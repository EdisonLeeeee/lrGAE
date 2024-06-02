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


class FusedBCE(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        
    def forward(self, left, right, paris, positive=True):
        out = self.decoder(left, right, paris)
        if positive:
            labels = torch.ones_like(out)
        else:
            labels = torch.zeros_like(out)
        loss = F.binary_cross_entropy(out, labels)   
        return loss
    
class SCELoss(nn.Module):
    def __init__(self, alpha=3):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x, y):
        return sce_loss(x, y, alpha=self.alpha)
    
class SIGLoss(nn.Module):
    def forward(self, x, y):
        return sig_loss(x, y)    

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss




def semi_loss(z1, z2, tau):
    def sim(z1, z2):
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())    

    f = lambda x: torch.exp(x / tau)

    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    
    loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    return loss


def uniformity_loss(features,t,max_size=30000,batch=10000):
    # calculate loss
    n = features.size(0)
    features = torch.nn.functional.normalize(features)
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