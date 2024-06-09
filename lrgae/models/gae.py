import torch
import torch.nn as nn
from torch_geometric.data import Data

from lrgae.losses import FusedBCE
from lrgae.utils import random_negative_sampler


class GAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = FusedBCE(decoder)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        return self.encoder(x, edge_index, **kwargs)

    def train_step(self, graph: Data) -> torch.Tensor:
        x, edge_index = graph.x, graph.edge_index
        z = self.encoder(x, edge_index)
        left = right = z[-1]

        loss = self.loss_fn(left, right, edge_index, positive=True)

        neg_edges = random_negative_sampler(
            num_nodes=graph.num_nodes,
            num_neg_samples=edge_index.size(1),
            device=edge_index.device,
        )

        loss += self.loss_fn(left, right, neg_edges, positive=False)
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
