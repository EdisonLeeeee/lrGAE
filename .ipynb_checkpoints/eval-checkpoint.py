import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score


def train_nodeclas(embedding, data, device='cpu', num_runs=10, learning_rate=1e-2,
                   weight_decay=5e-4, epochs=100, l2_normalize=False):
    @torch.no_grad()
    def test(loader):
        clf.eval()
        logits = []
        labels = []
        for nodes in loader:
            logits.append(clf(embedding[nodes]))
            labels.append(y[nodes])
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        logits = logits.argmax(1)
        return (logits == labels).float().mean().item()

    if data.num_nodes > 1e5:
        batch_size = 4096
    else:
        batch_size = 512
        
    train_loader = DataLoader(data.train_mask.nonzero().squeeze(), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data.test_mask.nonzero().squeeze(), batch_size=20000)
    val_loader = DataLoader(data.val_mask.nonzero().squeeze(), batch_size=20000)

    data = data.to(device)
    y = data.y.squeeze()
    
    if l2_normalize:
        embedding = F.normalize(embedding, p=2, dim=1)  # Cora, Citeseer, Pubmed    

    loss_fn = nn.CrossEntropyLoss()
    clf = nn.Linear(embedding.size(1), y.max().item() + 1).to(device)

    print('Start Training (Node Classification)...')
    results = []
    
    for run in range(1, num_runs+1):
        nn.init.xavier_uniform_(clf.weight.data)
        nn.init.zeros_(clf.bias.data)
        optimizer = torch.optim.Adam(clf.parameters(), 
                                     lr=learning_rate, 
                                     weight_decay=weight_decay)

        best_val_metric = test_metric = 0
        for epoch in tqdm(range(1, epochs+1), desc=f'Training on runs {run}...'):
            clf.train()
            for nodes in train_loader:
                optimizer.zero_grad()
                loss_fn(clf(embedding[nodes]), y[nodes]).backward()
                optimizer.step()
                
            val_metric, test_metric = test(val_loader), test(test_loader)
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test_metric
        results.append(best_test_metric)
        print(f'Runs {run}: accuracy {best_test_metric:.2%}')
                          
    print(f'Node Classification Results ({num_runs} runs):\n'
          f'Accuracy: {np.mean(results):.2%} ± {np.std(results):.2%}')
    
    return np.mean(results), np.std(results)
    
@torch.no_grad()
def eval_linkpred(z, decoder, pos_edges, neg_edges, batch_size):
    decoder.eval()
   
    pos_preds = []
    for perm in DataLoader(range(pos_edges.size(1)), batch_size):
        edges = pos_edges[:, perm]
        pos_preds += [decoder(z, edges, sigmoid=True).squeeze().cpu()]
    pos_preds = torch.cat(pos_preds, dim=0)

    neg_preds = []
    for perm in DataLoader(range(neg_edges.size(1)), batch_size):
        edges = neg_edges[:, perm]
        neg_preds += [decoder(z, edges, sigmoid=True).squeeze().cpu()]
    neg_preds = torch.cat(neg_preds, dim=0)

    y = torch.cat([
        pos_preds.new_ones(pos_preds.size(0)),
        neg_preds.new_zeros(neg_preds.size(0))
    ]).cpu().numpy()
    preds = torch.cat([pos_preds, neg_preds], dim=0).cpu().numpy()

    return roc_auc_score(y, preds), average_precision_score(y, preds)