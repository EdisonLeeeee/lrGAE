import torch
import torch.nn as nn
from torch_geometric.nn import Sequential

from lrgae.resolver import (activation_resolver, layer_resolver,
                            normalization_resolver)


class DotProductEdgeDecoder(nn.Module):
    """Dot-Product Edge Decoder"""

    def __init__(self, left=2, right=2, *args, **kwargs):
        super().__init__()
        self.left = left
        self.right = right

    def reset_parameters(self):
        return

    def forward(self, left, right, pairs, sigmoid=True):
        x = left[pairs[0]] * right[pairs[1]]
        x = x.sum(-1)

        if sigmoid:
            return x.sigmoid()
        else:
            return x


class EdgeDecoder(nn.Module):
    """MLP Edge Decoder"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels=1,
        num_layers=2,
        dropout=0.5,
        activation="relu",
        norm="none",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers        

        network = []
        for i in range(num_layers):
            is_last_layer = i == num_layers - 1
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if is_last_layer else hidden_channels
            layer = layer_resolver("linear", first_channels, second_channels)

            if not is_last_layer and dropout > 0:
                network.append((nn.Dropout(dropout), "x -> x"))
            network.append((layer, "x -> x"))
            if not is_last_layer and norm != "none":
                network.append(
                    (normalization_resolver(norm, second_channels), "x -> x")
                )
            if not is_last_layer and activation != "none":
                # whether to add last activation
                network.append((activation_resolver(activation), "x -> x"))

        self.network = Sequential("x", network)

    def reset_parameters(self):
        for layer in self.network:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, left, right, pairs, sigmoid=True):
        x = left[pairs[0]] * right[pairs[1]]
        x = self.network(x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x

class FeatureDecoder(nn.Module):
    """MLP Feature Decoder"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels=1,
        num_layers=2,
        dropout=0.5,
        activation="relu",
        norm="none",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        network = []
        for i in range(num_layers):
            is_last_layer = i == num_layers - 1
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if is_last_layer else hidden_channels
            layer = layer_resolver("linear", first_channels, second_channels)

            if not is_last_layer and dropout > 0:
                network.append((nn.Dropout(dropout), "x -> x"))
            network.append((layer, "x -> x"))
            
            if not is_last_layer and norm != "none":
                network.append(
                    (normalization_resolver(norm, second_channels), "x -> x")
                )
            if not is_last_layer and activation != "none":
                # whether to add last activation
                network.append((activation_resolver(activation), "x -> x"))

        self.network = Sequential("x", network)

    def reset_parameters(self):
        for layer in self.network:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x):
        x = self.network(x)
        return x
            
class CrossCorrelationDecoder(nn.Module):
    """Cross-Correlation Edge Decoder"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels=1,
        num_layers=2,
        dropout=0.5,
        activation="relu",
        norm="none",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        network = []
        for i in range(num_layers):
            is_last_layer = i == num_layers - 1
            first_channels = -1 if i == 0 else hidden_channels
            second_channels = out_channels if is_last_layer else hidden_channels
            layer = layer_resolver("linear", first_channels, second_channels)

            if not is_last_layer and dropout > 0:
                network.append((nn.Dropout(dropout), "x -> x"))
            network.append((layer, "x -> x"))
            if not is_last_layer and norm != "none":
                network.append(
                    (normalization_resolver(norm, second_channels), "x -> x")
                )
            if not is_last_layer and activation != "none":
                # whether to add last activation
                network.append((activation_resolver(activation), "x -> x"))

        self.network = Sequential("x", network)

    def reset_parameters(self):
        for layer in self.network:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, left, right, pairs, sigmoid=True):
        if not isinstance(left, (list, tuple)):
            left = [left]
        if not isinstance(right, (list, tuple)):
            right = [right]
        xs = []
        for l in left:
            for r in right:
                xs.append(l[pairs[0]] * r[pairs[1]])

        x = torch.cat(xs, dim=-1)
        x = self.network(x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x

class HeteroEdgeDecoder(nn.Module):
    def __init__(
        self,
        metadata,
        hidden_channels,
        out_channels=1,
        num_layers=2,
        dropout=0.5,
        activation="relu",
        norm="none",
    ):
        super().__init__()
        self.layers = nn.ModuleDict({
            '__'.join(edge_type): EdgeDecoder(-1, 
                                   hidden_channels=hidden_channels,
                                   out_channels=out_channels,
                                   num_layers=num_layers,
                                   dropout=dropout,
                                   activation=activation,
                                   norm=norm,
                                  )
            for edge_type in metadata[1]
        })       

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, left, right, pairs, sigmoid=True):
        out_dict = {}
        for edge_type, edge_index in pairs.items():
            src, _, dst = edge_type
            layer = self.layers['__'.join(edge_type)]
            out_dict[edge_type] = layer(left[src], right[dst], edge_index, sigmoid=sigmoid)
        return out_dict


class HeteroCrossCorrelationDecoder(nn.Module):
    def __init__(
        self,
        metadata,
        hidden_channels,
        out_channels=1,
        num_layers=2,
        dropout=0.5,
        activation="relu",
        norm="none",
    ):
        super().__init__()
        self.layers = nn.ModuleDict({
            '__'.join(edge_type): CrossCorrelationDecoder(-1, 
                                   hidden_channels=hidden_channels,
                                   out_channels=out_channels,
                                   num_layers=num_layers,
                                   dropout=dropout,
                                   activation=activation,
                                   norm=norm,
                                  )
            for edge_type in metadata[1]
        })       

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, left, right, pairs, sigmoid=True):
        out_dict = {}
        for edge_type, edge_index in pairs.items():
            src, _, dst = edge_type
            layer = self.layers['__'.join(edge_type)]
            left_list = [l[src] for l in left]
            right_list = [r[dst] for r in right]
            out_dict[edge_type] = layer(left_list, right_list, edge_index, sigmoid=sigmoid)
        return out_dict