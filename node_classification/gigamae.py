import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm.auto import tqdm

from torch.utils.data import DataLoader

# custom modules
from lrgae.dataset import get_dataset
from lrgae.utils import set_seed, tab_printer
from lrgae.encoders import GNNEncoder, PCA, Node2Vec
from lrgae.decoders import EdgeDecoder, CrossCorrelationDecoder, FeatureDecoder
from lrgae.masks import MaskFeature, NullMask
from lrgae.models import GiGaMAE

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
parser.add_argument('--seed', type=int, default=2024, help='Random seed for model and dataset. (default: 2024)')

parser.add_argument("--layer", nargs="?", default="gat", help="GNN layer, (default: gat)")
parser.add_argument("--encoder_activation", nargs="?", default="prelu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=512, help='Channels of hidden representation. (default: 512)')
parser.add_argument('--hidden_channels', type=int, default=512, help='Channels of hidden representation. (default: 512)')
parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.5, help='Dropout probability of encoder. (default: 0.8)')
parser.add_argument("--encoder_norm", nargs="?", default="none", help="Normalization (default: none)")

parser.add_argument('--decoder_channels', type=int, default=32, help='Channels of decoder layers. (default: 32)')
parser.add_argument("--decoder_activation", nargs="?", default="prelu", help="Activation function for GNN encoder, (default: prelu)")
parser.add_argument('--decoder_layers', type=int, default=1, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')
parser.add_argument("--decoder_norm", nargs="?", default="none", help="Normalization (default: none)")

parser.add_argument('--node_p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskNode')
parser.add_argument('--edge_p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge')
parser.add_argument("--replace_rate", type=float, default=0.0)
parser.add_argument("--alpha", type=float, default=3, help="`pow`coefficient for `sce` loss")

parser.add_argument('--embedding_dim', type=int, default = 64)
parser.add_argument('--walk_length', type=int, default = 5)
parser.add_argument('--context_size', type=int, default = 5)
parser.add_argument('--walks_per_node', type=int, default = 5)
parser.add_argument('--node2vec_batchsize', type=int, default = 512)
parser.add_argument('--node2vec_epoch', type=int, default = 20)
parser.add_argument('--node2vec_lr', type=float, default = 0.01)
parser.add_argument('--node2vec_neg_samples', type=int, default = 5)
parser.add_argument('--random_walk_p', type=float, default = 1.0)
parser.add_argument('--random_walk_q', type=float, default = 1.0)

parser.add_argument('--ratio', type=float, default=0.5)

parser.add_argument('--l1_e', type=int, default = 4)
parser.add_argument('--l2_e', type=int, default = 1)
parser.add_argument('--l12_e', type=int, default = 1)
parser.add_argument('--l1_f', type=int, default = 1)
parser.add_argument('--l2_f', type=int, default = 4)
parser.add_argument('--l12_f', type=int, default = 1)
parser.add_argument('--l1_b', type=int, default = 2)
parser.add_argument('--l2_b', type=int, default = 2)
parser.add_argument('--l12_b', type=int, default = -1)

parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training. (default: 0.001)')
parser.add_argument('--lrdec_1', type=float, default=0.8)
parser.add_argument('--lrdec_2', type=int, default=200)
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay for link prediction training. (default: 0)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')

parser.add_argument('--l2_normalize', action='store_true', help='Whether to use l2 normalize output embedding. (default: False)')
parser.add_argument('--nodeclas_lr', type=float, default=0.01, help='Learning rate for training. (default: 0.01)')
parser.add_argument('--nodeclas_weight_decay', type=float, default=5e-5, help='weight_decay for node classification training. (default: 1e-3)')

parser.add_argument('--epochs', type=int, default=1500, help='Number of training epochs. (default: 500)')
parser.add_argument('--runs', type=int, default=1, help='Number of runs. (default: 1)')
parser.add_argument('--eval_steps', type=int, default=50, help='(default: 50)')
parser.add_argument("--device", type=int, default=0)


def mask_feature(x, node_p):
    
    #maks node_p% node
    mask = torch.empty((x.size(0), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < node_p
    x = x.clone()
    x[mask,:] = 0
    
    return x, mask

def dropout_edge(edge_index, edge_p):
    #drop edge_p% edge
    def filter_adj(row, col, mask):
        return row[mask], col[mask]
        
    row, col = edge_index
    p = torch.zeros(edge_index.shape[1]).to(edge_index.device) + 1 - edge_p
    stay = torch.bernoulli(p).to(torch.bool)
    mask = ~stay
    row, col = filter_adj(row, col, stay)
    edge_index = torch.stack([row, col], dim=0)

    return edge_index.long(), mask
        
def main():

    try:
        args = parser.parse_args()
        print(tab_printer(args))
    except:
        parser.print_help()
        exit(0)

    set_seed(args.seed)

    if args.device < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # (!IMPORTANT) Specify the path to your dataset directory ##############
    root = '~/public_data/pyg_data' # my root directory
    # root = '../data/'
    ########################################################################
    transform = T.Compose([
        T.ToUndirected(),
        T.ToDevice(device),
        # T.NormalizeFeatures(),
    ])
    data = get_dataset(root, args.dataset, transform=transform)


    print("Node2Vec training...")
    node2vec = Node2Vec(data=data, embed_dim=args.embedding_dim,
                         walk_length=args.walk_length,
                         context_size=args.context_size,
                         walks_per_node=args.walks_per_node,
                         num_negative_samples=args.node2vec_neg_samples,
                         p=args.random_walk_p, q=args.random_walk_q).to(device)
    node2vec.train()
    node2vec.fit(epochs=args.node2vec_epoch,
                 lr=args.node2vec_lr,
                 batch_size=args.node2vec_batchsize,
                 device=device)
    node2vec_embeds = node2vec.get_embedding()
    
    print("PCA training...")
    pca = PCA().to(device)
    pca_embeds = pca(data.x, args.ratio)
    
    num_heads = 4
    encoder = GNNEncoder(in_channels=data.num_features, 
                         hidden_channels=args.encoder_channels//num_heads, 
                         out_channels=args.hidden_channels,
                         num_layers=args.encoder_layers, 
                         dropout=args.encoder_dropout,
                         norm=args.encoder_norm, 
                         layer=args.layer, 
                         num_heads=num_heads,
                         activation=args.encoder_activation)
    decoder = [FeatureDecoder(in_channels=args.hidden_channels,
                              hidden_channels=args.hidden_channels * 2,
                              out_channels=node2vec_embeds.size(1),
                              num_layers=args.decoder_layers, 
                              dropout=args.decoder_dropout,
                             ),
                FeatureDecoder(in_channels=args.hidden_channels,
                              hidden_channels=args.hidden_channels * 2,
                              out_channels=pca_embeds.size(1),
                              num_layers=args.decoder_layers, 
                               dropout=args.decoder_dropout,
                             ),
               FeatureDecoder(in_channels=args.hidden_channels,
                              hidden_channels=args.hidden_channels * 2,
                              out_channels=node2vec_embeds.size(1)+pca_embeds.size(1),
                              num_layers=args.decoder_layers, 
                              dropout=args.decoder_dropout,
                             )
    ]    
    
    model = GiGaMAE(encoder=encoder, decoder=decoder).to(device)
    print(model)

    best_metric = None
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)    
    pbar = tqdm(range(1, 1 + args.epochs))
    for epoch in pbar:

        model.train()

        optimizer.zero_grad()

        mask_x, mask_index_node_binary = mask_feature(data.x, args.node_p)
        mask_edge, mask_index_edge = dropout_edge(data.edge_index, args.edge_p)
        
        mask_edge_node = mask_index_edge * data.edge_index[0] 
        mask_index_edge_binary = torch.zeros(data.x.shape[0]).to(device) 
        mask_index_edge_binary[mask_edge_node] = 1
        mask_index_edge_binary = mask_index_edge_binary.to(bool)
        mask_both_node_edge = mask_index_edge_binary & mask_index_node_binary
        mask_index_node_binary_sole = mask_index_node_binary & (~mask_both_node_edge)
        mask_index_edge_binary_sole  = mask_index_edge_binary & (~mask_both_node_edge)

        loss = model.train_step(emb_node2vec=node2vec_embeds, emb_pca=pca_embeds,
                         mask_x=mask_x, mask_edge=mask_edge, mask_index_node=mask_index_node_binary_sole,
                         mask_index_edge=mask_index_edge_binary_sole, mask_both_node_edge=mask_both_node_edge)
        loss.backward()
        if args.grad_norm > 0:
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)     
        optimizer.step()
        pbar.set_description(f'Loss: {loss.item():.4f}')
        
        if epoch % args.eval_steps == 0:
            print(f'\nEvaluating on epoch {epoch}...')
            # with torch.no_grad():
            #     model.eval()
            #     embedding = model(data.x, data.edge_index)[-1]
            #     results = label_classification(embedding, data.y)
            results = model.eval_nodeclas(data,
                               lr=args.nodeclas_lr,
                               weight_decay=args.nodeclas_weight_decay,
                               l2_normalize=args.l2_normalize,
                            mode='last',
                               runs=args.runs,
                               device=device)
            if best_metric is None:
                best_metric = results
            for metric, value in results.items():
                print(f'Averaged {metric}: {value:.2%}')
                if best_metric[metric] < value:
                    best_metric = results
                
    for metric, value in best_metric.items():
        print(f'Best averaged {metric}: {value:.2%}')  

if __name__ == "__main__":
    main()
