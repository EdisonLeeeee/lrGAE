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
from lrgae.loss import FusedBCE, info_nce_loss, log_rank_loss, hinge_auc_loss, auc_loss, semi_loss, SCELoss, uniformity_loss, simcse_loss


def random_negative_sampler(num_nodes, num_neg_samples, device):
    neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(device)
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
        mask,
        loss="bce",
        left=2,
        right=2,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask = mask
        self.left = left[0] if isinstance(left, (list, tuple)) and len(left) == 1 else left
        self.right = right[0] if isinstance(right, (list, tuple)) and len(right) == 1 else right
        
        if loss == "bce":
            self.loss_fn = FusedBCE(decoder)
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
            self.loss_fn = SCELoss(2)            
        else:
            raise ValueError(loss)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        return self.encoder(x, edge_index, **kwargs)

    def train_step_feature(self, graph):
        remaining_graph, masked_graph = self.mask(graph)
        remaining_features, remaining_edge_index = remaining_graph.x, remaining_graph.edge_index
        masked_features, masked_edge_index = masked_graph.x, masked_graph.edge_index
        mask = masked_graph.mask

        if self.left > 0:
            zA = self.encoder(remaining_features, remaining_edge_index)
        else:
            zA = [remaining_features]
            
        if self.right > 0:
            zB = self.encoder(masked_features, masked_edge_index)
        else:
            zB = [masked_features]

        left = self.decoder(zA[self.left], remaining_edge_index)[-1] if self.left > 0 else zA[self.left]
        right = self.decoder(zB[self.right], masked_edge_index)[-1] if self.right > 0 else zB[self.right]
        # left = self.decoder(zA[self.left], remaining_edge_index)[-1]
        # right = masked_features        
        loss = self.loss_fn(left.masked_select(mask), right.masked_select(mask))     
        return loss
        
    def train_step(self, graph):
        remaining_graph, masked_graph = self.mask(graph)
        x, remaining_edges = remaining_graph.x, remaining_graph.edge_index
        masked_edges = masked_graph.edge_index
        z = self.encoder(x, remaining_edges)
        
        if isinstance(self.left, (list, tuple)):
            left = [z[l] for l in self.left]
        else:
            left = z[self.left]
        if isinstance(self.right, (list, tuple)):
            right = [z[r] for r in self.right]
        else:
            right = z[self.right]

        loss = self.loss_fn(left, right, masked_edges, positive=True)

        neg_edges = random_negative_sampler(
            num_nodes=remaining_graph.num_nodes,
            num_neg_samples=masked_edges.size(1),
            device=masked_edges.device,
        )
        
        loss += self.loss_fn(left, right, neg_edges, positive=False)
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
        return y.cpu(), pred.cpu()
        

    def eval_nodeclas(self, 
                          data,
                          batch_size=512,
                          epochs=100,
                          runs=1,
                          lr=0.01,
                          weight_decay=0.,
                          l2_normalize=False,
                          mode='cat',
                          device='cpu'):

        with torch.no_grad():
            self.eval()
            embedding = self(data.x.to(device), data.edge_index.to(device))[1:]
            if mode == 'cat':
                embedding = torch.cat(embedding, dim=-1)
            else:
                embedding = embedding[-1]
            if l2_normalize:
                embedding = F.normalize(embedding, p=2, dim=1)  # Cora, Citeseer, Pubmed     

        train_x, train_y = embedding[data.train_mask], data.y[data.train_mask]
        val_x, val_y = embedding[data.val_mask], data.y[data.val_mask]
        test_x, test_y = embedding[data.test_mask], data.y[data.test_mask]
        results = linear_probing(train_x, 
                   train_y, 
                    val_x,
                   val_y,
                   test_x,
                   test_y,
                  lr=lr,
                  weight_decay=weight_decay,
                   batch_size=batch_size,
                                 runs=runs,
                                 epochs=epochs,
                   device=device)
        
        return {'acc': np.mean(results)}

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
        
        y, pred = self.test_step(data, 
                              data.pos_edge_label_index, 
                              data.neg_edge_label_index, 
                              batch_size=batch_size)
        
        return {'auc': roc_auc_score(y, pred),
                'ap': average_precision_score(y, pred)}

class MaskGAE(lrGAE):
    def __init__(
        self,
        encoder,
        decoder,
        mask,
        degree_decoder,
        loss="bce",
    ):
        super().__init__(encoder=encoder, decoder=decoder, mask=mask, loss=loss)
        self.degree_decoder = degree_decoder

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.degree_decoder.reset_parameters()
        
    def train_step(self, graph, alpha=0.):
        remaining_graph, masked_graph = self.mask(graph)
        x, remaining_edges = remaining_graph.x, remaining_graph.edge_index
        masked_edges = masked_graph.edge_index
        z = self.encoder(x, remaining_edges)
        left = right = z[-1]

        loss = self.loss_fn(left, right, masked_edges, positive=True)

        neg_edges = random_negative_sampler(
            num_nodes=remaining_graph.num_nodes,
            num_neg_samples=masked_edges.size(1),
            device=masked_edges.device,
        )
        
        loss += self.loss_fn(left, right, neg_edges, positive=False)
        if alpha > 0:
            deg = degree(masked_edges[1].flatten(), graph.num_nodes).float()
            deg = (deg - deg.mean()) / (deg.std() + 1e-6)
            loss += alpha * F.mse_loss(self.degree_decoder(left).squeeze(), deg)            
        return loss
        
class GraphMAE(lrGAE):
    def __init__(self, encoder, decoder, mask, neck, in_channels,
                 replace_rate=0.2, mask_rate=0.5,
                 alpha=1):
        super().__init__(encoder=encoder, decoder=decoder, mask=mask, loss='sce')
        self.encoder = encoder
        self.neck = neck
        self.decoder = decoder
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_channels))
        
        self.replace_rate = replace_rate
        self.mask_token_rate = 1 - self.replace_rate
        self.mask_rate = mask_rate

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.neck.reset_parameters()
        self.decoder.reset_parameters()
        self.enc_mask_token.data.zero_()
        
    def train_step(self, data):
        x, edge_index = data.x, data.edge_index
        device = x.device
        num_nodes = x.size(0)
        perm = torch.randperm(num_nodes, device=device)

        num_mask_nodes = int(self.mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=device)
            token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0  
            
        out_x[token_nodes] += self.enc_mask_token

        enc_rep = self.encoder(out_x, edge_index)      
        # enc_rep = torch.cat(enc_rep[1:], dim=1)
        enc_rep = enc_rep[-1]
        rep = self.neck(enc_rep)
        
        # if decoder_type not in ("mlp", "linear"):
        # * remask, re-mask
        rep[mask_nodes] = 0    
        recon = self.decoder(rep, edge_index)[-1]
        
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        
        loss = self.loss_fn(x_rec, x_init)
        return loss

class GraphMAE2(lrGAE):
    def __init__(self, encoder, decoder, mask, neck,
                 in_channels, num_hidden, num_remasking,
                 replace_rate=0.2, mask_rate=0.5,
                 remask_rate=0.5, remask_method="random",
                 alpha=1, lambd=1, momentum=0.996,):
        super().__init__(encoder=encoder, decoder=decoder, mask=mask, loss='sce')
        self.encoder = encoder
        self.decoder = decoder
        self.neck = neck
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_channels))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, num_hidden))
        
        self.num_remasking = num_remasking
        self.replace_rate = replace_rate
        self.mask_token_rate = 1 - self.replace_rate
        self.mask_rate = mask_rate
        self.remask_rate = remask_rate
        self.remask_method = remask_method
        self.lambd = lambd
        self.momentum = momentum

        self.input_loss = SCELoss(alpha=alpha)
        self.latent_loss = SCELoss(alpha=1)

        self.projector = nn.Sequential(
            nn.Linear(num_hidden, 256),
            nn.PReLU(),
            nn.Linear(256, num_hidden),
        )
        self.predictor = nn.Sequential(
            nn.PReLU(),
            nn.Linear(num_hidden, num_hidden)
        )

        self.encoder_ema = copy.deepcopy(self.encoder)
        self.projector_ema = copy.deepcopy(self.projector)

        # stop-gradient
        for p in self.encoder_ema.parameters():
            p.requires_grad = False
            # p.detach_()
        for p in self.projector_ema.parameters():
            p.requires_grad = False
            # p.detach_()

        self.reset_parameters_for_token()

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.neck.weight, gain=1.414)

    def train_step(self, data):
        x, edge_index = data.x, data.edge_index
        device = x.device
        num_nodes = x.size(0)  
        perm = torch.randperm(num_nodes, device=device)
        targets = torch.arange(x.size(0), device=device)

        # random masking
        num_mask_nodes = int(self.mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token
        
        enc_rep = self.encoder(out_x, edge_index)[-1]

        with torch.no_grad():
            # pass unmasked graph through the target generator to produce target representation
            latent_target = self.encoder_ema(x, edge_index)[-1]
            if targets is not None:
                latent_target = self.projector_ema(latent_target[targets])
            else:
                latent_target = self.projector_ema(latent_target[keep_nodes])

        # prediction and target
        if targets is not None:
            latent_pred = self.projector(enc_rep[targets])
            latent_pred = self.predictor(latent_pred)
            loss_latent = self.latent_loss(latent_pred, latent_target)
        else:
            latent_pred = self.projector(enc_rep[keep_nodes])
            latent_pred = self.predictor(latent_pred)
            loss_latent = self.latent_loss(latent_pred, latent_target)

        # ---- feature reconstruction ----
        origin_rep = self.neck(enc_rep)

        loss_rec_all = 0
        if self.remask_method == "random":
            for i in range(self.num_remasking):
                rep = origin_rep.clone()
                rep, remask_nodes, rekeep_nodes = self.random_remask(x, rep)
                recon = self.decoder(rep, edge_index)[-1]

                x_init = x[mask_nodes]
                x_rec = recon[mask_nodes]

                loss_rec = self.input_loss(x_init, x_rec)
                loss_rec_all += loss_rec

        elif self.remask_method == "fixed":
            origin_rep[mask_nodes] = 0
            x_rec = self.decoder(origin_rep, edge_index)[-1][mask_nodes]
            x_init = x[mask_nodes]
            loss_rec_all = self.input_loss(x_init, x_rec)

        loss = loss_latent + self.lambd * loss_rec_all

        self.ema_update()

        return loss

    def ema_update(self):
        def update(student, teacher):
            with torch.no_grad():
            # m = momentum_schedule[it]  # momentum parameter
                m = self.momentum
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        update(self.encoder, self.encoder_ema)
        update(self.projector, self.projector_ema)

    def random_remask(self, x, rep):
        
        device = x.device
        num_nodes = x.size(0)
        perm = torch.randperm(num_nodes, device=device)
        
        num_mask_nodes = int(self.mask_rate * num_nodes)
        remask_nodes = perm[: num_mask_nodes]
        rekeep_nodes = perm[num_mask_nodes: ]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token

        return rep, remask_nodes, rekeep_nodes



def linear_probing(train_x, 
                   train_y, 
                    val_x,
                   val_y,
                   test_x,
                   test_y,
                  lr=0.01,
                  weight_decay=0.,
                   batch_size=512,
                   runs=5,
                   epochs=100,
                   device='cpu'):

    @torch.no_grad()
    def test(loader):
        classifier.eval()
        logits = []
        labels = []
        for x, y in loader:
            logits.append(classifier(x))
            labels.append(y)
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        logits = logits.argmax(1)
        return (logits == labels).float().mean().item()
        
    train_y = train_y.squeeze().to(device)
    val_y = val_y.squeeze().to(device)
    test_y = test_y.squeeze().to(device)
    
    num_classes = train_y.max().item() + 1
    classifier = nn.Linear(train_x.size(1), num_classes).to(device)        

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=20000)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=20000)

    results = []
    for _ in range(runs):
        nn.init.xavier_uniform_(classifier.weight.data)
        nn.init.zeros_(classifier.bias.data)
        optimizer = torch.optim.Adam(classifier.parameters(), 
                                     lr=lr, 
                                     weight_decay=weight_decay)
    
        best_val_metric = test_metric = 0
        for epoch in range(1, epochs + 1):
            classifier.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                F.cross_entropy(classifier(x), y).backward()
                optimizer.step()
                
            val_metric, test_metric = test(val_loader), test(test_loader)
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test_metric
        results.append(best_test_metric)
            
    return results