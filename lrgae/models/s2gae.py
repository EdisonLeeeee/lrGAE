from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData

from lrgae.losses import FusedBCE, HeteroFusedBCE
from lrgae.negative_sampling import negative_sampling


class S2GAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        mask,
        negative_sampler = 'random',
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask = mask
        if decoder.__class__.__name__.startswith('Hetero'):
            self.loss_fn = HeteroFusedBCE(decoder)
        else:
            self.loss_fn = FusedBCE(decoder)
        assert negative_sampler in ['random', 'similarity', 'degree', 'hard_negative']
        self.negative_sampler = negative_sampler

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        return self.encoder(x, edge_index, **kwargs)

    def train_step_homo(self, graph: Data) -> torch.Tensor:
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

    def train_step_hetero(self, graph: HeteroData) -> torch.Tensor:
        remaining_graph, masked_graph = self.mask(graph)
        z = self.encoder(remaining_graph.x_dict, remaining_graph.edge_index_dict)
        left = right = z[1:]

        neg_edge_index_dict = {}
        for edge_type, masked_edges in masked_graph.edge_index_dict.items():
            src, _, dst = edge_type
            neg_edge_index_dict[edge_type] = negative_sampling(self.negative_sampler,
                                      x=(masked_graph[src].x, masked_graph[dst].x), 
                                      edge_index=masked_edges,
                                      num_neg_samples=masked_edges.size(1),
                                      left=left,
                                      right=right,  
                                      decoder=self.decoder,
                                     )   
        loss = self.loss_fn(left, right, masked_graph.edge_index_dict, neg_edge_index_dict)
        return loss

    def train_step(self, graph: Union[Data, HeteroData]) -> torch.Tensor:
        if isinstance(graph, Data):
            return self.train_step_homo(graph)
        else:
            return self.train_step_hetero(graph)    

