import numpy as np
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import add_self_loops, negative_sampling, degree
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# custom modules
from loss import info_nce_loss, ce_loss, log_rank_loss, hinge_auc_loss, auc_loss, semi_loss, SCELoss, uniformity_loss, simcse_loss


def random_negative_sampler(edge_index, num_nodes, num_neg_samples):
    neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index)
    return neg_edges

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class lrGAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        loss="ce",
        left=2,
        right=2,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.left = left[0] if isinstance(left, (list, tuple)) and len(left) == 1 else left
        self.right = right[0] if isinstance(right, (list, tuple)) and len(right) == 1 else right
        
        if loss == "ce":
            self.loss_fn = ce_loss
        elif loss == "auc":
            self.loss_fn = auc_loss
        elif loss == "info_nce":
            self.loss_fn = info_nce_loss
        elif loss == "log_rank":
            self.loss_fn = log_rank_loss
        elif loss == "hinge_auc":
            self.loss_fn = hinge_auc_loss
        elif loss == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss == 'sce':
            self.loss_fn = SCELoss()            
        else:
            raise ValueError(loss)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        return self.encoder(x, edge_index, **kwargs)

    def train_step_feature(self, remaining_graph, masked_graph):

        remaining_features, edge_index = remaining_graph.x, remaining_graph.edge_index
        masked_features = masked_graph.x
        mask = masked_graph.get('mask')
        
        if mask is None:
            # for plain GAE
            mask = masked_features.new_ones(masked_features.size(0), 1, dtype=torch.bool)

        if self.left > 0:
            zA = self.encoder(remaining_features, edge_index)
        else:
            zA = [remaining_features]
            
        if self.right > 0:
            zB = self.encoder(masked_features, edge_index)
        else:
            zB = [masked_features]

        left = self.decoder(zA[self.left], edge_index)[-1] if self.left > 0 else zA[self.left]
        right = self.decoder(zB[self.right], edge_index)[-1] if self.right > 0 else zB[self.right]
        loss = self.loss_fn(left.masked_select(mask), right.masked_select(mask))     
        return loss
        
    def train_step(self, remaining_graph, masked_graph):

        x, remaining_edges = remaining_graph.x, remaining_graph.edge_index
        masked_edges = masked_graph.edge_index
        z = self.encoder(x, remaining_edges)
        aug_edge_index, _ = add_self_loops(remaining_edges)
        neg_edges = random_negative_sampler(
            aug_edge_index,
            num_nodes=remaining_graph.num_nodes,
            num_neg_samples=masked_edges.size(1),
        )
        
        if isinstance(self.left, (list, tuple)):
            left = [z[l] for l in self.left]
        else:
            left = z[self.left]
        if isinstance(self.right, (list, tuple)):
            right = [z[r] for r in self.right]
        else:
            right = z[self.right]
            
        pos_out = self.decoder(left, right, masked_edges, sigmoid=False)
        neg_out = self.decoder(left, right, neg_edges, sigmoid=False)
        loss = self.loss_fn(pos_out, neg_out)        
        return loss

    @torch.no_grad()
    def batch_predict(self, left, right, edges, batch_size=2**16):
        preds = []
        for edge in DataLoader(TensorDataset(edges.t()), batch_size=batch_size):
            edge = edge[0].t()
            preds.append(self.decoder(left, right, edge).squeeze())
        pred = torch.cat(preds, dim=0)
        return pred.cpu()
        
    @torch.no_grad()
    def test_step(self, data, pos_edge_index, neg_edge_index, batch_size=2**16):
        self.eval()
        z = self(data.x, data.edge_index)
        left = z[self.left]
        right = z[self.right]
        pos_pred = self.batch_predict(left, right, pos_edge_index)
        neg_pred = self.batch_predict(left, right, neg_edge_index)

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))

        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.cpu().numpy(), pred.cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


    def eval_nodeclas(self, 
                          data,
                          batch_size=512,
                          epochs=100,
                          runs=10,
                          lr=0.01,
                          weight_decay=0.,
                          l2_normalize=False,
                          mode='cat',
                          device='cpu'):
        
        @torch.no_grad()
        def test(loader):
            classifier.eval()
            logits = []
            labels = []
            for nodes in loader:
                logits.append(classifier(embedding[nodes]))
                labels.append(node_labels[nodes])
            logits = torch.cat(logits, dim=0).cpu()
            labels = torch.cat(labels, dim=0).cpu()
            logits = logits.argmax(1)
            return (logits == labels).float().mean().item()
            
        with torch.no_grad():
            self.eval()
            embedding = self(data.x.to(device), data.edge_index.to(device))[1:]
            if mode == 'cat':
                embedding = torch.cat(embedding, dim=-1)
            else:
                embedding = embedding[-1]
            if l2_normalize:
                embedding = F.normalize(embedding, p=2, dim=1)  # Cora, Citeseer, Pubmed     
                
        node_labels = data.y.squeeze().to(device)
        num_classes = node_labels.max().item() + 1
        loss_fn = nn.CrossEntropyLoss()
        classifier = nn.Linear(embedding.size(1), num_classes).to(device)        

        train_loader = DataLoader(data.train_mask.nonzero().squeeze(), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(data.test_mask.nonzero().squeeze(), batch_size=20000)
        val_loader = DataLoader(data.val_mask.nonzero().squeeze(), batch_size=20000)
        
        results = []
        for run in range(1, runs+1):
            nn.init.xavier_uniform_(classifier.weight.data)
            nn.init.zeros_(classifier.bias.data)
            optimizer = torch.optim.Adam(classifier.parameters(), 
                                         lr=lr, 
                                         weight_decay=weight_decay)
    
            best_val_metric = test_metric = 0
            for epoch in range(1, epochs + 1):
                classifier.train()
                for nodes in train_loader:
                    optimizer.zero_grad()
                    loss_fn(classifier(embedding[nodes]), node_labels[nodes]).backward()
                    optimizer.step()
                    
                val_metric, test_metric = test(val_loader), test(test_loader)
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_test_metric = test_metric
            results.append(best_test_metric)
            print(f'Runs {run}: accuracy {best_test_metric:.2%}')
        
        return np.mean(results), np.std(results)      

    def eval_linkpred(self, 
                      data,
                      batch_size=2**16,
                      epochs=100,
                      runs=10,
                      lr=0.01,
                      weight_decay=0.,
                      l2_normalize=False,
                      mode='cat',
                      device='cpu'):
        
        return self.test_step(data, data.pos_edge_label_index, data.neg_edge_label_index, batch_size=batch_size)