from typing import Any, Optional, Union

import torch
from torch import nn, Tensor
from torch_geometric.nn import (GATConv, GATv2Conv, GCNConv, GINConv, Linear,
                                SAGEConv)
from torch_geometric.resolver import resolver


def swish(x: Tensor) -> Tensor:
    return x * x.sigmoid()

def activation_resolver(query: Optional[Union[Any, str]] = 'relu', *args, **kwargs):
    if query is None or query == 'none':
        return torch.nn.Identity()
    base_cls = torch.nn.Module
    base_cls_repr = 'Act'
    acts = [
        act for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, base_cls)
    ]
    acts += [
        swish,
    ]
    act_dict = {}
    return resolver(acts, act_dict, query, base_cls, base_cls_repr, *args,
                    **kwargs)
    
def normalization_resolver(query: Optional[Union[Any, str]], *args, **kwargs):
    if query is None or query == 'none':
        return torch.nn.Identity()    
    import torch_geometric.nn.norm as norm
    base_cls = torch.nn.Module
    base_cls_repr = 'Norm'
    norms = [
        norm for norm in vars(norm).values()
        if isinstance(norm, type) and issubclass(norm, base_cls)
    ]
    norm_dict = {}
    return resolver(norms, norm_dict, query, base_cls, base_cls_repr, *args,
                    **kwargs)

def layer_resolver(name, first_channels, second_channels, heads=1):
    if name == "sage":
        layer = SAGEConv(first_channels, second_channels)
    elif name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    elif name == "gin":
        layer = GINConv(nn.Sequential(Linear(first_channels, second_channels), 
                                      nn.LayerNorm(second_channels),
                                      nn.PReLU(),
                                      Linear(second_channels, second_channels),
                                      # nn.LayerNorm(second_channels),
                                     ), train_eps=True)
    elif name == "gat":
        layer = GATConv(-1, second_channels, heads=heads)
    elif name == "gat2":
        layer = GATv2Conv(-1, second_channels, heads=heads)
    elif name == 'linear':
        layer = Linear(first_channels, second_channels)
    else:
        raise ValueError(name)
    return layer