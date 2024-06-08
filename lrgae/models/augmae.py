import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from lrgae.losses import SCELoss, uniformity_loss


class AUGMAE(nn.Module):
    def __init__(self, encoder, decoder, neck, uniformity_layer,
                 alpha=1, replace_rate=0., mask_rate=0.5,):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.neck = neck
        self.uniformity_layer = uniformity_layer
        self.enc_mask_token = nn.Parameter(torch.zeros(1, encoder.in_channels))

        self.criterion = SCELoss(alpha=alpha)
        self.mask_criterion = SCELoss(alpha=alpha)

        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.mask_token_rate = 1 - replace_rate

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.neck.reset_parameters()
        self.decoder.reset_parameters()
        self.uniformity_layer.reset_parameters()
        self.enc_mask_token.data.zero_()


    def forward(self, x, edge_index, **kwargs):
        return self.encoder(x, edge_index, **kwargs)
        
    def train_step(self, graph: Data, alpha_adv, mask_prob, lamda, belta) -> torch.Tensor:
        x, edge_index = graph.x, graph.edge_index
        device = x.device
        num_nodes = x.size(0)

        # random masking
        perm = torch.randperm(num_nodes, device=device)
        num_random_mask_nodes = int(
            self.mask_rate * num_nodes * (1. - alpha_adv))
        random_mask_nodes = perm[: num_random_mask_nodes]
        random_keep_nodes = perm[num_random_mask_nodes:]

        # adversarial masking
        mask_ = mask_prob[:, 1]
        perm_adv = torch.randperm(num_nodes, device=device)
        adv_keep_token = perm_adv[:int(num_nodes * (1. - alpha_adv))]
        mask_[adv_keep_token] = 1.
        Mask_ = mask_.reshape(-1, 1)

        adv_keep_nodes = mask_.nonzero().reshape(-1)
        adv_mask_nodes = (1 - mask_).nonzero().reshape(-1)

        mask_nodes = torch.cat(
            (random_mask_nodes, adv_mask_nodes), dim=0).unique()
        keep_nodes = torch.tensor(np.intersect1d(
            random_keep_nodes.cpu().numpy(),
            adv_keep_nodes.cpu().numpy())).to(device)

        num_mask_nodes = mask_nodes.shape[0]

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
            out_x = out_x * Mask_
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            out_x = out_x * Mask_
            token_nodes = mask_nodes
            out_x[token_nodes] = 0.0

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
        lamda = lamda * (1 - alpha_adv)

        # if args.dataset in ("IMDB-BINARY", "IMDB-MULTI", "PROTEINS", "MUTAG","REDDIT-BINERY", "COLLAB"):
        #     # sub_g = use_g.subgraph(keep_nodes)
        #     # enc_rep_sub_g = enc_rep[keep_nodes]
        #     # graph_emb = pooler(sub_g,enc_rep_sub_g)
        #     # graph_emb = F.relu(self.uniformity_layer(graph_emb))
        #     # u_loss = uniformity_loss(graph_emb,lamda)
        #     pass
        # else:
        node_eb = F.relu(self.uniformity_layer(enc_rep))
        u_loss = uniformity_loss(node_eb, lamda)

        loss = self.criterion(x_rec, x_init) + u_loss
        num_all_noeds = mask_prob[:, 1].sum() + mask_prob[:, 0].sum()
        loss_mask = -self.mask_criterion(x_rec, x_init) + \
            belta * (torch.tensor([1.]).to(device) /
                     torch.sin(torch.pi / num_all_noeds * (mask_prob[:, 0].sum())))
        return loss, loss_mask
