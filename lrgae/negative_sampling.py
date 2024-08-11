import torch
import torch.nn.functional as F
from torch_geometric.utils import degree

NUM_CANDIDATES = 20

def negative_sampling(method, 
                      x, edge_index, 
                      num_neg_samples, 
                      left,
                      right,
                      decoder,
                      num_nodes=None):
    if not isinstance(x, tuple):
        x = (x, x)    
    num_nodes = num_nodes or (x[0].size(0), x[1].size(0))
    device = x[0].device
    
    if method == 'similarity':
        neg_edges = similarity_negative_sampler(
            x=x,
            num_nodes=num_nodes,
            num_neg_samples=num_neg_samples,
            device=device,
        )
        
    elif method == 'random':
        neg_edges = random_negative_sampler(
            num_nodes=num_nodes,
            num_neg_samples=num_neg_samples,
            device=device,
        )
    elif method == 'degree':
        neg_edges = degree_negative_sampler(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_neg_samples,
            device=device,
        )   
    elif method == 'hard_negative':
        neg_edges = hard_negative_sampler(
            x=(left, right),
            decoder=decoder,
            num_nodes=num_nodes,
            num_neg_samples=num_neg_samples,
            device=device,
        )                
    else:
        raise ValueErrpr(f'Unknown negative sampler {method}')
    return neg_edges
    
def random_negative_sampler(num_nodes, num_neg_samples, device):
    src = torch.randint(0, num_nodes[0], size=(num_neg_samples,), device=device)
    dst = torch.randint(0, num_nodes[1], size=(num_neg_samples,), device=device)
    neg_edges = torch.stack([src, dst], dim=0)
    return neg_edges
    
def degree_negative_sampler(edge_index, num_nodes, num_neg_samples, device):
    candidates = random_negative_sampler(num_nodes, num_neg_samples=num_neg_samples*NUM_CANDIDATES, device=device)
    d = degree(edge_index[1], num_nodes)
    row, col = candidates
    score = (d[row] - d[col]).abs()
    k = score.topk(num_neg_samples, largest=False).indices
    neg_edges = candidates[:, k]
    return neg_edges
    
def similarity_negative_sampler(x, num_nodes, num_neg_samples, device):
    left, right = x
    candidates = random_negative_sampler(num_nodes, num_neg_samples=num_neg_samples*NUM_CANDIDATES, device=device)
    row, col = candidates
    score = F.cosine_similarity(left[row], right[col])    
    k = score.topk(num_neg_samples, largest=False).indices
    neg_edges = candidates[:, k]
    return neg_edges

def hard_negative_sampler(x, decoder, num_nodes, num_neg_samples, device):
    left, right = x
    candidates = random_negative_sampler(num_nodes, num_neg_samples=num_neg_samples*NUM_CANDIDATES, device=device)
    row, col = candidates
    with torch.no_grad():
        score = decoder(left, right, candidates).squeeze()
    k = score.topk(num_neg_samples, largest=False).indices
    neg_edges = candidates[:, k]
    return neg_edges

