
import torch
import torch.nn as nn
from torch_geometric.data import Data

from lrgae.losses import semi_loss


class GiGaMAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        if isinstance(decoder, (list, tuple)):
            decoder = nn.ModuleList(decoder)
        self.decoder = decoder

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for module in self.decoder:
            module.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        return self.encoder(x, edge_index, **kwargs)

    def train_step(self,
                   emb_node2vec,
                   emb_pca,
                   mask_x,
                   mask_edge,
                   mask_index_node,
                   mask_index_edge,
                   mask_both_node_edge,
                   tau=0.7,
                   l1_e=4, l2_e=2, l12_e=6,
                   l1_f=2, l2_f=5, l12_f=6,
                   l1_b=2, l2_b=3, l12_b=3,
                   ) -> torch.Tensor:

        z = self.encoder(mask_x, mask_edge)[-1]
        p1, p2, p3 = self.decoder
        recon_z1, recon_z2, recon_z3 = p1(z), p2(z), p3(z)

        loss1_f = semi_loss(
            emb_node2vec[mask_index_node], recon_z1[mask_index_node], tau)
        loss1_e = semi_loss(
            emb_node2vec[mask_index_edge], recon_z1[mask_index_edge], tau)
        loss1_both = semi_loss(
            emb_node2vec[mask_both_node_edge], recon_z1[mask_both_node_edge], tau)

        loss2_f = semi_loss(emb_pca[mask_index_node],
                            recon_z2[mask_index_node], tau)
        loss2_e = semi_loss(emb_pca[mask_index_edge],
                            recon_z2[mask_index_edge], tau)
        loss2_both = semi_loss(
            emb_pca[mask_both_node_edge], recon_z2[mask_both_node_edge], tau)

        loss12_f = semi_loss(torch.cat((emb_node2vec, emb_pca), 1)[
                             mask_index_node], recon_z3[mask_index_node], tau)
        loss12_e = semi_loss(torch.cat((emb_node2vec, emb_pca), 1)[
                             mask_index_edge], recon_z3[mask_index_edge], tau)
        loss12_both = semi_loss(torch.cat((emb_node2vec, emb_pca), 1)[
                                mask_both_node_edge], recon_z3[mask_both_node_edge], tau)

        loss_e = l1_e * loss1_e + l2_e * loss2_e + l12_e * loss12_e
        loss_f = l1_f * loss1_f + l2_f * loss2_f + l12_f * loss12_f
        loss_both = l1_b * loss1_both + l2_b * loss2_both + l12_b * loss12_both

        info_loss = loss_e.mean() + loss_f.mean() + loss_both.mean()

        return info_loss
