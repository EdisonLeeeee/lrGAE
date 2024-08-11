import torch
import torch.nn.functional as F
from torch import nn


def cosine_similarity(z1, z2):
    assert z1.size() == z2.size()
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    return torch.mm(z1, z2.t())

def simcse_loss(a, b, tau=0.02):
    similarity = cosine_similarity(a, b) / tau
    labels = torch.arange(
        similarity.size(0), dtype=torch.long, device=similarity.device
    )
    return F.cross_entropy(similarity, labels)
    
def auc_loss(pos_out, neg_out):
    return torch.square(1 - (pos_out - neg_out)).sum()

def hinge_auc_loss(pos_out, neg_out):
    return (torch.square(torch.clamp(1 - (pos_out - neg_out), min=0))).sum()

def log_rank_loss(pos_out, neg_out, num_neg=1):
    return -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-15).mean()

def info_nce_loss(pos_out, neg_out):
    pos_exp = torch.exp(pos_out)
    neg_exp = torch.sum(torch.exp(neg_out), 1, keepdim=True)
    return -torch.log(pos_exp / (pos_exp + neg_exp) + 1e-15).mean()


class SimCSE(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        
    def forward(self, left, right, pairs, neg_pairs):
        left = left[pairs[0]]
        right = right[pairs[0]]
        loss = simcse_loss(left, right)
        return loss
        
class FusedNCE(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        
    def forward(self, left, right, pairs, neg_pairs):
        pos_out = self.decoder(left, right, pairs)
        neg_out = self.decoder(left, right, neg_pairs)
        loss = info_nce_loss(pos_out, neg_out)   
        return loss

class FusedLogRankAUC(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        
    def forward(self, left, right, pairs, neg_pairs):
        pos_out = self.decoder(left, right, pairs)
        neg_out = self.decoder(left, right, neg_pairs)
        loss = log_rank_loss(pos_out, neg_out)   
        return loss
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
        
class FusedHingeAUC(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        
    def forward(self, left, right, pairs, neg_pairs):
        pos_out = self.decoder(left, right, pairs)
        neg_out = self.decoder(left, right, neg_pairs)
        loss = hinge_auc_loss(pos_out, neg_out)   
        return loss
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
        
class FusedAUC(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        
    def forward(self, left, right, pairs, neg_pairs):
        pos_out = self.decoder(left, right, pairs)
        neg_out = self.decoder(left, right, neg_pairs)
        loss = auc_loss(pos_out, neg_out)   
        return loss
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
        

class FusedBCE(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        
    def forward(self, left, right, pairs, neg_pairs=None):
        pos_out = self.decoder(left, right, pairs)
        labels = torch.ones_like(pos_out)
        loss = F.binary_cross_entropy(pos_out, labels)   
        
        if neg_pairs is not None:
            neg_out = self.decoder(left, right, neg_pairs)
            neg_labels = torch.zeros_like(neg_out)
            loss += F.binary_cross_entropy(neg_out, neg_labels)   
        return loss
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class HeteroFusedBCE(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        
    def forward(self, left, right, pairs, neg_pairs=None):
        loss = 0.
        out = self.decoder(left, right, pairs)
        for edge_type, pos_out in out.items():
            labels = torch.ones_like(pos_out)
            loss += F.binary_cross_entropy(pos_out, labels)   
        
        if neg_pairs is not None:
            out = self.decoder(left, right, neg_pairs)
            for edge_type, neg_out in out.items():
                neg_labels = torch.zeros_like(neg_out)
                loss += F.binary_cross_entropy(neg_out, neg_labels)   
        return loss
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
        
class SCELoss(nn.Module):
    def __init__(self, alpha=3):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, left, right):
        x = F.normalize(left, p=2, dim=-1)
        y = F.normalize(right, p=2, dim=-1)
    
        loss = (1 - (x * y).sum(dim=-1)).pow_(self.alpha)
    
        loss = loss.mean()        
        return loss
    
class SIGLoss(nn.Module):
    def forward(self, left, right):
        x = F.normalize(left, p=2, dim=-1)
        y = F.normalize(right, p=2, dim=-1)
        loss = (x * y).sum(1)
        loss = torch.sigmoid(-loss)
        loss = loss.mean()
        return loss

class NT_Xent(nn.Module):
    def __init__(self, tau=0.5):
        super().__init__()
        self.tau = tau
        
    def forward(self, left, right):
        f = lambda x: torch.exp(x / self.tau)
    
        refl_sim = f(cosine_similarity(left, left))
        between_sim = f(cosine_similarity(left, right))
        
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        return loss
        
        
def semi_loss(z1, z2, tau):
    f = lambda x: torch.exp(x / tau)

    refl_sim = f(cosine_similarity(z1, z1))
    between_sim = f(cosine_similarity(z1, z2))
    
    loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    return loss


def uniformity_loss(features,t,max_size=30000,batch=10000):
    # calculate loss
    n = features.size(0)
    features = F.normalize(features, p=2, dim=1)
    if n < max_size:
        loss = torch.log(torch.exp(2.*t*((features@features.T)-1.)).mean())
    else:
        total_loss = 0.
        permutation = torch.randperm(n)
        features = features[permutation]
        for i in range(0, n, batch):
            batch_features = features[i:i + batch]
            batch_loss = torch.log(torch.exp(2.*t*((batch_features@batch_features.T)-1.)).mean())
            total_loss += batch_loss
        loss = total_loss / (n // batch)
    return loss