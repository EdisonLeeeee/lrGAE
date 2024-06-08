import torch
import torch.nn as nn
from torch_geometric.data import Data

from lrgae.losses import SCELoss


class GraphMAE(nn.Module):
    def __init__(self, encoder, decoder, neck,
                 replace_rate=0.2, mask_rate=0.5,
                 alpha=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.neck = neck
        self.enc_mask_token = nn.Parameter(torch.zeros(1, encoder.in_channels))

        self.replace_rate = replace_rate
        self.mask_token_rate = 1 - self.replace_rate
        self.mask_rate = mask_rate
        self.loss_fn = SCELoss(alpha=alpha)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.neck.reset_parameters()
        self.decoder.reset_parameters()
        self.enc_mask_token.data.zero_()

    def forward(self, x, edge_index, **kwargs):
        return self.encoder(x, edge_index, **kwargs)

    def train_step(self, graph: Data) -> torch.Tensor:
        x, edge_index = graph.x, graph.edge_index
        device = x.device
        num_nodes = x.size(0)
        perm = torch.randperm(num_nodes, device=device)

        num_mask_nodes = int(self.mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=device)
            token_nodes = mask_nodes[perm_mask[: int(
                self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(
                self.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=device)[
                :num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        enc_rep = self.encoder(out_x, edge_index)
        # enc_rep = torch.cat(enc_rep[1:], dim=1)
        enc_rep = enc_rep[-1]
        rep = self.neck(enc_rep)

        # if decoder_type not in ("mlp", "linear"):
        # * remask, re-mask
        rep[mask_nodes] = 0
        recon = self.decoder(rep, edge_index)[-1]

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.loss_fn(x_rec, x_init)
        return loss
