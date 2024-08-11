import torch
import torch.nn as nn
from torch_geometric.data import Data

from lrgae.losses import FusedBCE
from lrgae.negative_sampling import negative_sampling


class S2GAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        mask,
        negative_sampler = 'hard_negative',
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask = mask
        self.loss_fn = FusedBCE(decoder)
        assert negative_sampler in ['random', 'similarity', 'degree', 'hard_negative']
        self.negative_sampler = negative_sampler

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        return self.encoder(x, edge_index, **kwargs)

    def train_step(self, graph: Data) -> torch.Tensor:
        remaining_graph, masked_graph = self.mask(graph)
        x, remaining_edge_index = remaining_graph.x, remaining_graph.edge_index
        masked_edges = remaining_graph.masked_edges

        z = self.encoder(x, remaining_edge_index)

        left = right = z[1:]
        neg_edges = negative_sampling(self.negative_sampler,
                                      x=graph.x, 
                                      edge_index=graph.edge_index,
                                      num_neg_samples=masked_edges.size(1),
                                      left=left,
                                      right=right,  
                                      decoder=self.decoder,
                                     )
        loss = self.loss_fn(left, right, masked_edges, neg_edges)
        return loss
