import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import (
    Node2Vec,
    Sequential,
)
from torch_sparse import SparseTensor
from resolver import activation_resolver, normalization_resolver, layer_resolver


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
            block.append((nn.Dropout(dropout), 'x -> x'))
            block.append((conv, 'x, edge_index -> x'))
            if not is_last_layer or (is_last_layer and self.add_last_bn):
                # whether to add last BN
                if norm != 'none':
                    block.append((normalization_resolver(norm, second_channels*heads), 'x -> x'))
            if not is_last_layer or (is_last_layer and self.add_last_act):
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
        edge_index = to_sparse_tensor(edge_index, x.size(0))
        out = [x]
        for block in self.network:
            x = block(x, edge_index)
            out.append(x)
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(TransformerEncoder, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class PCA(nn.Module):
    def __init__(self):
        super().__init__()

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


class NodeToVec(nn.Module):
    def __init__(self, data, embed_dim, walk_length,
                 context_size, walks_per_node,
                 num_negative_samples, p=1., q=1.):
        super().__init__()
        self.data = data
        self.node2vec = Node2Vec(edge_index=data.edge_index,
                                 embedding_dim=embed_dim,
                                 walk_length=walk_length,
                                 context_size=context_size,
                                 walks_per_node=walks_per_node,
                                 num_negative_samples=num_negative_samples,
                                 p=p, q=q, sparse=True)

    def forward(self):
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

    def target_generation(self):
        self.eval()
        return self.node2vec().detach()


class AdversMask(nn.Module):
    def __init__(self, generator, fc_in_channels, fc_out_channels):
        super().__init__()
        self.generator = generator
        self.fc = nn.Linear(fc_in_channels, fc_out_channels)

    def forward(self, x, edge_index):
        x = self.generator(x, edge_index)[-1]
        z = F.gumbel_softmax(self.fc(x), hard=True)

        return z
