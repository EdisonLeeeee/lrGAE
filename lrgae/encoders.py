import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Node2Vec as n2v
from torch_geometric.nn import Sequential
from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, HeteroConv, Sequential
from torch_geometric.nn.module_dict import ModuleDict

from lrgae.resolver import (activation_resolver, layer_resolver,
                            normalization_resolver)


def to_sparse_tensor(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)


class GNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels=None,
        num_heads=4,
        num_layers=2,
        dropout=0.5,
        norm='batchnorm',
        layer="gcn",
        activation="elu",
        add_last_act=True,
        add_last_bn=True,
    ):

        super().__init__()

        out_channels = out_channels or hidden_channels
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers        

        self.add_last_act = add_last_act
        self.add_last_bn = add_last_bn

        networks = []
        for i in range(num_layers):
            is_last_layer = i == num_layers - 1
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if is_last_layer else hidden_channels
            heads = 1 if i == num_layers - 1 or 'gat' not in layer else num_heads
            conv = layer_resolver(layer, first_channels,
                                  second_channels, heads)

            block = []
            if dropout > 0:
                block.append((nn.Dropout(dropout), 'x -> x'))
            block.append((conv, 'x, edge_index -> x'))
            if not is_last_layer or (is_last_layer and add_last_bn):
                # whether to add last BN
                if norm != 'none':
                    block.append((normalization_resolver(norm, second_channels*heads), 'x -> x'))
            if not is_last_layer or (is_last_layer and add_last_act):
                # whether to add last activation
                if activation != 'none':
                    block.append((activation_resolver(activation), 'x -> x'))
            networks.append(Sequential('x, edge_index', block))

        self.network = nn.Sequential(*networks)

    def reset_parameters(self):
        for block in self.network:
            for layer in block:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self, x, edge_index):
        edge_index = to_sparse_tensor(edge_index, num_nodes=x.size(0))
        out = [x]
        for block in self.network:
            x = block(x, edge_index)
            out.append(x)
        return out

class PCA(nn.Module):
    def svd_flip(self, u, v):
        # columns of u, rows of v
        max_abs_cols = torch.argmax(torch.abs(u), 0)
        i = torch.arange(u.shape[1]).to(u.device)
        signs = torch.sign(u[max_abs_cols, i])
        u *= signs
        v *= signs.view(-1, 1)
        return u, v

    def postprocess(self, embs, ratio):
        # PCA
        mu = torch.mean(embs, dim=0, keepdim=True)
        X = embs - mu
        U, S, V = torch.svd(X)
        U, Vt = self.svd_flip(U, V)
        accumulate, sum_S = 0.0, sum(S.detach().cpu().tolist())
        for idx, s in enumerate(S.detach().cpu().tolist(), 1):
            accumulate += s / sum_S
            if accumulate > ratio:
                break
        X = torch.mm(X, Vt[:idx].T)

        # whitening
        u, s, vt = torch.svd(torch.mm(X.T, X) / (X.shape[0] - 1.0))
        W = torch.mm(u, torch.diag(1.0 / torch.sqrt(s)))
        X = torch.mm(X, W)
        return X

    def forward(self, embs, ratio):
        if embs.shape[0] > embs.shape[1]:
            pca_emb = self.postprocess(embs, ratio)
        else:
            embs1 = embs[:, 0:embs.shape[1]//2]
            embs2 = embs[:, embs.shape[1]//2:]

            pca_emb1 = self.postprocess(embs1, ratio)
            pca_emb2 = self.postprocess(embs2, ratio)

            pca_emb = torch.cat((pca_emb1, pca_emb2), 1)

        return pca_emb


class Node2Vec(nn.Module):
    def __init__(self, data, embed_dim, walk_length,
                 context_size, walks_per_node,
                 num_negative_samples, p=1., q=1.):
        super().__init__()
        self.data = data
        self.node2vec = n2v(edge_index=data.edge_index,
                                 embedding_dim=embed_dim,
                                 walk_length=walk_length,
                                 context_size=context_size,
                                 walks_per_node=walks_per_node,
                                 num_negative_samples=num_negative_samples,
                                 p=p, q=q, sparse=True)

    @torch.no_grad()
    def get_embedding(self):
        self.node2vec.eval()
        return self.node2vec()

    def fit(self, epochs, lr, batch_size, device="cpu"):
        loader = self.node2vec.loader(
            batch_size=batch_size, shuffle=True, num_workers=0)
        optimizer = torch.optim.SparseAdam(
            list(self.node2vec.parameters()), lr=lr)

        for epoch in range(epochs):
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = self.node2vec.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()

class HeteroGNNEncoder(nn.Module):
    def __init__(
        self,
        metadata,
        hidden_channels,
        out_channels=None,
        num_heads=4,
        num_layers=2,
        dropout=0.5,
        norm='batchnorm',
        layer="sage",
        activation="elu",
        add_last_act=True,
        add_last_bn=True,
    ):

        super().__init__()

        out_channels = out_channels or hidden_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers  
        self.add_last_act = add_last_act
        self.add_last_bn = add_last_bn
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        
        for i in range(num_layers):
            is_last_layer = i == num_layers - 1
            first_channels = -1 if i == 0 else hidden_channels
            second_channels = out_channels if is_last_layer else hidden_channels
            heads = 1 if i == num_layers - 1 or 'gat' not in layer else num_heads
            conv = HeteroConv({
                edge_type: layer_resolver(layer, first_channels,
                                  second_channels, heads)
                for edge_type in metadata[1]
            })
            if not is_last_layer or (is_last_layer and add_last_bn):
                norm_layer = nn.ModuleDict(
                    {
                        node_type: normalization_resolver(norm, second_channels*heads)
                        for node_type in metadata[0]
                    }
                )
                self.norms.append(norm_layer)
            if not is_last_layer or (is_last_layer and add_last_act):
                self.acts.append(activation_resolver(activation))                
            self.convs.append(conv)

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
                    
    def forward(self, x_dict, edge_index_dict):
        xs = [x_dict]
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                key: self.dropout(self.acts[i](self.norms[i][key](x)))
                for key, x in x_dict.items()
            }
            xs.append(x_dict)
        return xs