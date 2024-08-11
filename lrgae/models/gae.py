import torch
import torch.nn as nn
from torch_geometric.data import Data

from lrgae.losses import FusedBCE
from lrgae.negative_sampling import negative_sampling


class GAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        negative_sampler = 'random',
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = FusedBCE(decoder)
        assert negative_sampler in ['random', 'similarity', 'degree', 'hard_negative']
        self.negative_sampler = negative_sampler

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        return self.encoder(x, edge_index, **kwargs)

    def train_step(self, graph: Data) -> torch.Tensor:
        x, edge_index = graph.x, graph.edge_index
        z = self.encoder(x, edge_index)
        left = right = z[-1]

        neg_edges = negative_sampling(self.negative_sampler,
                                      x=x, 
                                      edge_index=edge_index,
                                      num_neg_samples=edge_index.size(1),
                                      left=left,
                                      right=right,  
                                      decoder=self.decoder,
                                     )
        loss = self.loss_fn(left, right, edge_index, neg_edges)
        return loss
        


class GAE_f(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = nn.MSELoss()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        return self.encoder(x, edge_index, **kwargs)

    def train_step(self, graph: Data) -> torch.Tensor:
        x, edge_index = graph.x, graph.edge_index
        z = self.encoder(x, edge_index)
        left = self.decoder(z[-1])
        right = x

        loss = self.loss_fn(left, right)
        return loss
