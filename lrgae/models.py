import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import degree

# custom modules
from lrgae.loss import (FusedBCE, SCELoss, auc_loss, hinge_auc_loss,
                        info_nce_loss, log_rank_loss, semi_loss, simcse_loss,
                        uniformity_loss)


def random_negative_sampler(num_nodes, num_neg_samples, device):
    neg_edges = torch.randint(0, num_nodes,
                              size=(2, num_neg_samples)).to(device)
    return neg_edges


class lrGAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        mask=None,
        loss="bce",
        left=2,
        right=2,
        view='AA',
    ):
        super().__init__()
        assert view in ['AA', 'AB', 'BB']
        self.encoder = encoder
        self.decoder = decoder
        self.mask = mask
        self.left = left
        self.right = right
        self.view = view

        if loss == "bce":
            self.loss_fn = FusedBCE(decoder)
        elif loss == "auc":
            self.loss_fn = auc_loss
        elif loss == "info_nce":
            self.loss_fn = info_nce_loss
        elif loss == "log_rank":
            self.loss_fn = log_rank_loss
        elif loss == "hinge_auc":
            self.loss_fn = hinge_auc_loss
        elif loss == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss == 'sce':
            self.loss_fn = SCELoss(2)
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

    def train_step(self, graph):
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

        loss = self.loss_fn(left, right, masked_edges, positive=True)

        neg_edges = random_negative_sampler(
            num_nodes=graph.num_nodes,
            num_neg_samples=masked_edges.size(1),
            device=masked_edges.device,
        )

        loss += self.loss_fn(left, right, neg_edges, positive=False)
        return loss

    def extra_repr(self) -> str:
        return f'Left={self.left}, Right={self.right}, View={self.view},'


class MaskGAE(lrGAE):
    def __init__(
        self,
        encoder,
        decoder,
        mask,
        degree_decoder,
        loss="bce",
    ):
        super().__init__(encoder=encoder, decoder=decoder, mask=mask,
                         loss=loss, left=encoder.num_layers, right=encoder.num_layers,)
        self.degree_decoder = degree_decoder

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.degree_decoder.reset_parameters()

    def train_step(self, graph, alpha=0.):
        remaining_graph, masked_graph = self.mask(graph)
        x, remaining_edge_index = remaining_graph.x, remaining_graph.edge_index
        masked_edges = remaining_graph.masked_edges

        z = self.encoder(x, remaining_edge_index)
        left = right = z[-1]

        loss = self.loss_fn(left, right, masked_edges, positive=True)

        neg_edges = random_negative_sampler(
            num_nodes=graph.num_nodes,
            num_neg_samples=masked_edges.size(1),
            device=masked_edges.device,
        )

        loss += self.loss_fn(left, right, neg_edges, positive=False)
        if alpha > 0:
            deg = degree(masked_edges[1].flatten(), graph.num_nodes).float()
            deg = (deg - deg.mean()) / (deg.std() + 1e-6)
            loss += alpha * \
                F.mse_loss(self.degree_decoder(left).squeeze(), deg)
        return loss


class S2GAE(lrGAE):
    def __init__(
        self,
        encoder,
        decoder,
        mask,
        loss="bce",
    ):
        encoder_layers = encoder.num_layers
        super().__init__(encoder=encoder, decoder=decoder, mask=mask, loss=loss, left=list(
            range(1, encoder_layers + 1)), right=list(range(1, encoder_layers + 1)))

    def train_step(self, graph):
        remaining_graph, masked_graph = self.mask(graph)
        x, remaining_edge_index = remaining_graph.x, remaining_graph.edge_index
        masked_edges = remaining_graph.masked_edges
        z = self.encoder(x, remaining_edge_index)
        left = right = z[1:]

        loss = self.loss_fn(left, right, masked_edges, positive=True)

        neg_edges = random_negative_sampler(
            num_nodes=graph.num_nodes,
            num_neg_samples=masked_edges.size(1),
            device=masked_edges.device,
        )

        loss += self.loss_fn(left, right, neg_edges, positive=False)
        return loss


class GraphMAE(lrGAE):
    def __init__(self, encoder, decoder, neck,
                 replace_rate=0.2, mask_rate=0.5,
                 alpha=1):
        super().__init__(encoder=encoder, decoder=decoder,
                         loss='sce', left=encoder.num_layers, right=0)
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

    def train_step(self, graph):
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


class AUGMAE(lrGAE):
    def __init__(self, encoder, decoder, neck, uniformity_layer,
                 alpha=1, replace_rate=0., mask_rate=0.5,):
        super().__init__(encoder=encoder, decoder=decoder, mask=None,
                         loss='sce', left=encoder.num_layers, right=0)
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

    def train_step(self, graph, alpha_adv, mask_prob, lamda, belta):
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


class GraphMAE2(lrGAE):
    def __init__(self, encoder, decoder, neck, num_remasking,
                 replace_rate=0.2, mask_rate=0.5,
                 remask_rate=0.5, remask_method="random",
                 alpha=1, lambd=1, momentum=0.996,):
        super().__init__(encoder=encoder, decoder=decoder, mask=None,
                         loss='sce', left=encoder.num_layers, right=0)
        self.neck = neck
        self.enc_mask_token = nn.Parameter(torch.zeros(1, encoder.in_channels))
        self.dec_mask_token = nn.Parameter(
            torch.zeros(1, encoder.out_channels))

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

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.neck.weight, gain=1.414)

    def train_step(self, graph):
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


class GiGaMAE(lrGAE):
    def __init__(self, encoder, decoder):
        super().__init__(encoder=encoder, decoder=decoder, left=encoder.num_layers, right=0)
        self.p1, self.p2, self.p3 = decoder[0], decoder[1], decoder[2]

    def train_step(self, emb_node2vec, emb_pca,
                   mask_x, mask_edge, mask_index_node, mask_index_edge, mask_both_node_edge,
                   tau=0.7,
                   l1_e=4, l2_e=2, l12_e=6,
                   l1_f=2, l2_f=5, l12_f=6,
                   l1_b=2, l2_b=3, l12_b=3,
                   ):
        z = self.encoder(mask_x, mask_edge)[-1]
        recon_z1, recon_z2, recon_z3 = self.p1(z), self.p2(z), self.p3(z)

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
        loss_both = l1_b * loss1_both + l1_b * loss2_both + l12_b * loss12_both

        info_loss = loss_e.mean() + loss_f.mean() + loss_both.mean()

        return info_loss


def linear_probing_cv(x, y, test_ratio=0.1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import OneHotEncoder, normalize
    from sklearn.svm import SVC

    def prob_to_one_hot(y_pred):
        ret = np.zeros(y_pred.shape, bool)
        indices = np.argmax(y_pred, axis=1)
        for i in range(y_pred.shape[0]):
            ret[i][indices[i]] = True
        return ret

    x = x.cpu().numpy()
    y = y.cpu().numpy().reshape(-1, 1)

    onehot_encoder = OneHotEncoder(categories='auto').fit(y)
    y = onehot_encoder.transform(y).toarray().astype(bool)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_ratio)
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=4, cv=5,
                       verbose=0)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    print(micro)
    return [micro]


def linear_probing(train_x,
                   train_y,
                   val_x,
                   val_y,
                   test_x,
                   test_y,
                   lr=0.01,
                   weight_decay=0.,
                   batch_size=512,
                   runs=5,
                   epochs=100,
                   device='cpu'):

    @torch.no_grad()
    def test(loader):
        classifier.eval()
        logits = []
        labels = []
        for x, y in loader:
            logits.append(classifier(x))
            labels.append(y)
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        logits = logits.argmax(1)
        return (logits == labels).float().mean().item()

    train_y = train_y.squeeze().to(device)
    val_y = val_y.squeeze().to(device)
    test_y = test_y.squeeze().to(device)

    num_classes = train_y.max().item() + 1
    classifier = nn.Linear(train_x.size(1), num_classes).to(device)

    train_loader = DataLoader(TensorDataset(
        train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=20000)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=20000)

    results = []
    for _ in range(runs):
        nn.init.xavier_uniform_(classifier.weight.data)
        nn.init.zeros_(classifier.bias.data)
        optimizer = torch.optim.Adam(classifier.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)

        best_val_metric = test_metric = 0
        for epoch in range(1, epochs + 1):
            classifier.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                F.cross_entropy(classifier(x), y).backward()
                optimizer.step()

            val_metric, test_metric = test(val_loader), test(test_loader)
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test_metric
        results.append(best_test_metric)

    return results
