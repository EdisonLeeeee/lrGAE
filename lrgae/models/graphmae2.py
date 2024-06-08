import copy
import torch
import torch.nn as nn
from torch_geometric.data import Data

from lrgae.losses import SCELoss


class GraphMAE2(nn.Module):
    def __init__(self, encoder, decoder, neck, num_remasking,
                 replace_rate=0.2, mask_rate=0.5,
                 remask_rate=0.5, remask_method="random",
                 alpha=1, lambd=1, momentum=0.996,):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.neck = neck
        self.enc_mask_token = nn.Parameter(torch.zeros(1, encoder.in_channels))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, encoder.out_channels))  # noqa

        self.num_remasking = num_remasking
        self.replace_rate = replace_rate
        self.mask_token_rate = 1 - self.replace_rate
        self.mask_rate = mask_rate
        self.remask_rate = remask_rate
        self.remask_method = remask_method
        self.lambd = lambd
        self.momentum = momentum

        self.input_loss = SCELoss(alpha=alpha)
        self.latent_loss = SCELoss(alpha=1)

        self.projector = nn.Sequential(
            nn.Linear(encoder.out_channels, 256),
            nn.PReLU(),
            nn.Linear(256, encoder.out_channels),
        )
        self.predictor = nn.Sequential(
            nn.PReLU(),
            nn.Linear(encoder.out_channels, encoder.out_channels)
        )

        self.encoder_ema = copy.deepcopy(self.encoder)
        self.projector_ema = copy.deepcopy(self.projector)

        # stop-gradient
        for p in self.encoder_ema.parameters():
            p.requires_grad = False
            # p.detach_()
        for p in self.projector_ema.parameters():
            p.requires_grad = False
            # p.detach_()

        self.reset_parameters_for_token()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.neck.reset_parameters()
        self.decoder.reset_parameters()
        self.reset_parameters_for_token()

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.neck.weight, gain=1.414)

    def forward(self, x, edge_index, **kwargs):
        return self.encoder(x, edge_index, **kwargs)

    def train_step(self, graph: Data) -> torch.Tensor:
        x, edge_index = graph.x, graph.edge_index
        device = x.device
        num_nodes = x.size(0)
        perm = torch.randperm(num_nodes, device=device)
        targets = torch.arange(x.size(0), device=device)

        # random masking
        num_mask_nodes = int(self.mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token

        enc_rep = self.encoder(out_x, edge_index)[-1]

        with torch.no_grad():
            # pass unmasked graph through the target generator to produce target representation
            latent_target = self.encoder_ema(x, edge_index)[-1]
            if targets is not None:
                latent_target = self.projector_ema(latent_target[targets])
            else:
                latent_target = self.projector_ema(latent_target[keep_nodes])

        # prediction and target
        if targets is not None:
            latent_pred = self.projector(enc_rep[targets])
            latent_pred = self.predictor(latent_pred)
            loss_latent = self.latent_loss(latent_pred, latent_target)
        else:
            latent_pred = self.projector(enc_rep[keep_nodes])
            latent_pred = self.predictor(latent_pred)
            loss_latent = self.latent_loss(latent_pred, latent_target)

        # ,-- feature reconstruction,--
        origin_rep = self.neck(enc_rep)

        loss_rec_all = 0
        if self.remask_method == "random":
            for i in range(self.num_remasking):
                rep = origin_rep.clone()
                rep, remask_nodes, rekeep_nodes = self.random_remask(x, rep)
                recon = self.decoder(rep, edge_index)[-1]

                x_init = x[mask_nodes]
                x_rec = recon[mask_nodes]

                loss_rec = self.input_loss(x_init, x_rec)
                loss_rec_all += loss_rec

        elif self.remask_method == "fixed":
            origin_rep[mask_nodes] = 0
            x_rec = self.decoder(origin_rep, edge_index)[-1][mask_nodes]
            x_init = x[mask_nodes]
            loss_rec_all = self.input_loss(x_init, x_rec)

        loss = loss_latent + self.lambd * loss_rec_all

        self.ema_update()

        return loss

    def ema_update(self):
        def update(student, teacher):
            with torch.no_grad():
                # m = momentum_schedule[it]  # momentum parameter
                m = self.momentum
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        update(self.encoder, self.encoder_ema)
        update(self.projector, self.projector_ema)

    def random_remask(self, x, rep):

        device = x.device
        num_nodes = x.size(0)
        perm = torch.randperm(num_nodes, device=device)

        num_mask_nodes = int(self.mask_rate * num_nodes)
        remask_nodes = perm[: num_mask_nodes]
        rekeep_nodes = perm[num_mask_nodes:]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token

        return rep, remask_nodes, rekeep_nodes
