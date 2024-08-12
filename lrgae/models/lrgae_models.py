from typing import Union

import torch
import torch.nn as nn
from torch_geometric.data import Data, HeteroData

from lrgae.losses import FusedBCE, SCELoss, FusedAUC, FusedNCE, SimCSE, HeteroFusedBCE
from lrgae.negative_sampling import negative_sampling


class lrGAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        mask,
        loss: str = "bce",
        left: int = 2,
        right: int = 2,
        view: str = 'AA',
        pair: str = 'vv',
        negative_sampler = 'random',
        **kwargs,
    ):
        super().__init__()
        assert view in ['AA', 'AB', 'BB']
        assert pair in ['vv', 'vu']
        assert negative_sampler in ['random', 'similarity', 'degree', 'hard_negative']
        self.negative_sampler = negative_sampler    
        
        if pair == 'vv':
            self.train_step = self.train_step_feature
        else:
            self.train_step = self.train_step_structure

        self.encoder = encoder
        self.decoder = decoder
        self.mask = mask
        self.left = left
        self.right = right
        self.view = view

        if loss == "bce":
            if decoder.__class__.__name__.startswith('Hetero'):
                self.loss_fn = HeteroFusedBCE(decoder)
            else:
                self.loss_fn = FusedBCE(decoder)
            assert pair == 'vu'
        elif loss == 'auc':
            self.loss_fn = FusedAUC(decoder)
            assert pair == 'vu'
        elif loss == 'nce':
            self.loss_fn = FusedNCE(decoder)
            assert pair == 'vu'        
        elif loss == 'simcse':
            self.loss_fn = SimCSE(decoder)            
        elif loss == 'mse':
            self.loss_fn = nn.MSELoss()
            assert pair == 'vv'
        elif loss == 'sce':
            self.loss_fn = SCELoss(alpha=kwargs.get('alpha', 2.0))
            assert pair == 'vv'
        else:
            raise ValueError(loss)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        return self.encoder(x, edge_index, **kwargs)

    def train_step_feature(self, graph):
        remaining_graph, masked_graph = self.mask(graph)

        # case A=B
        if self.view == 'AA':
            masked_graph = remaining_graph
        elif self.view == 'BB':
            remaining_graph = masked_graph

        remaining_features, remaining_edge_index = remaining_graph.x, remaining_graph.edge_index
        masked_features, masked_edge_index = masked_graph.x, masked_graph.edge_index
        # in most cases `remaining_edge_index` == `masked_edge_index` == `graph.edge_index`
        masked_nodes = remaining_graph.masked_nodes

        if self.left > 0:
            zA = self.encoder(remaining_features, remaining_edge_index)
        else:
            zA = [remaining_features]

        if self.view == 'AB':
            if self.right > 0:
                zB = self.encoder(masked_features, masked_edge_index)
            else:
                zB = [masked_features]
        else:
            zB = zA

        # TODO: deal with non-GNN decoder which does not return a list
        left = self.decoder(
            zA[self.left], remaining_edge_index)[-1] if self.left > 0 else zA[self.left]
        right = self.decoder(
            zB[self.right], masked_edge_index)[-1] if self.right > 0 else zB[self.right]
        loss = self.loss_fn(left.masked_select(masked_nodes),
                            right.masked_select(masked_nodes))
        return loss

    def train_step_structure(self, graph: Union[Data, HeteroData]) -> torch.Tensor:
        if isinstance(graph, Data):
            return self.train_step_structure_homo(graph)
        else:
            return self.train_step_structure_hetero(graph)
            
    def train_step_structure_homo(self, graph: Data) -> torch.Tensor:
        remaining_graph, masked_graph = self.mask(graph)

        # case A=B
        if self.view == 'AA':
            masked_graph = remaining_graph
        elif self.view == 'BB':
            remaining_graph = masked_graph

        remaining_features, remaining_edge_index = remaining_graph.x, remaining_graph.edge_index
        masked_features, masked_edge_index = masked_graph.x, masked_graph.edge_index
        # in most cases here `masked_features` == `remaining_features` == `graph.x`
        masked_edges = remaining_graph.masked_edges

        zA = self.encoder(remaining_features, remaining_edge_index)

        if self.left > 0:
            zA = self.encoder(remaining_features, remaining_edge_index)
        else:
            zA = [remaining_features]

        if self.view == 'AB':
            if self.right > 0:
                zB = self.encoder(masked_features, masked_edge_index)
            else:
                zB = [masked_features]
        else:
            zB = zA

        left = zA[self.left]
        right = zB[self.right]

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

    def train_step_structure_hetero(self, graph: HeteroData) -> torch.Tensor:
        remaining_graph, masked_graph = self.mask(graph)

        # case A=B
        if self.view == 'AA':
            masked_graph = remaining_graph
        elif self.view == 'BB':
            remaining_graph = masked_graph

        if self.left > 0:
            zA = self.encoder(remaining_graph.x_dict, remaining_graph.edge_index_dict)
        else:
            zA = [remaining_graph.x_dict]

        if self.view == 'AB':
            if self.right > 0:
                zB = self.encoder(masked_graph.x_dict, masked_graph.edge_index_dict)
            else:
                zB = [remaining_graph.x_dict]
        else:
            zB = zA

        left = zA[self.left]
        right = zB[self.right]

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

    def extra_repr(self) -> str:
        return f'l={self.left}, r={self.right}, view={self.view}, pair={self.pair}'
