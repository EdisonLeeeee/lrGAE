import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.data import Data

from lrgae.losses import FusedBCE
from lrgae.utils import random_negative_sampler


class MaskGAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        mask,
        degree_decoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.degree_decoder = degree_decoder
        self.mask = mask
        self.loss_fn = FusedBCE(decoder)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.degree_decoder.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        return self.encoder(x, edge_index, **kwargs)

    def train_step(self, graph: Data, alpha: float = 0.) -> torch.Tensor:
        remaining_graph, masked_graph = self.mask(graph)
        x, remaining_edge_index = remaining_graph.x, remaining_graph.edge_index
        masked_edges = remaining_graph.masked_edges

        z = self.encoder(x, remaining_edge_index)
        left = right = z[-1]

        loss = self.loss_fn(left, right, masked_edges, positive=True)

        neg_edges = random_negative_sampler(
            num_nodes=graph.num_nodes,
            num_neg_samples=masked_edges.size(1),
            device=masked_edges.device,
        )

        loss += self.loss_fn(left, right, neg_edges, positive=False)
        if alpha > 0:
            deg = degree(masked_edges[1].flatten(), graph.num_nodes).float()
            deg = (deg - deg.mean()) / (deg.std() + 1e-6)
            loss += alpha * \
                F.mse_loss(self.degree_decoder(left).squeeze(), deg)
        return loss
