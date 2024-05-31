import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm.auto import tqdm

from torch.utils.data import DataLoader

# custom modules
from dataset import get_dataset
from utils import set_seed, tab_printer
from encoders import GNNEncoder
from decoders import EdgeDecoder, CrossCorrelationDecoder, FeatureDecoder
from masks import MaskEdge, MaskPath, NullMask
from model import lrGAE

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
parser.add_argument("--mask", nargs="?", default="path", help="Masking stractegy, `path`, `edge` or `None` (default: path)")
parser.add_argument('--seed', type=int, default=2022, help='Random seed for model and dataset. (default: 2022)')

parser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of hidden representation. (default: 128)')
parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.8)')
parser.add_argument("--encoder_norm", nargs="?", default="none", help="Normalization (default: none)")

parser.add_argument('--decoder_channels', type=int, default=32, help='Channels of decoder layers. (default: 32)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')
parser.add_argument("--decoder_norm", nargs="?", default="none", help="Normalization (default: none)")

parser.add_argument('--left', nargs='+', type=int, default=2, help='Left layer. (default: 2)')
parser.add_argument('--right', nargs='+', type=int, default=2, help='Right layer. (default: 2)')

parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training. (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for link prediction training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')

parser.add_argument("--start", nargs="?", default="node", help="Which Type to sample starting nodes for random walks, (default: node)")
parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge/MaskPath')

parser.add_argument('--l2_normalize', action='store_true', help='Whether to use l2 normalize output embedding. (default: False)')
parser.add_argument('--nodeclas_lr', type=float, default=0.01, help='Learning rate for training. (default: 0.01)')
parser.add_argument('--nodeclas_weight_decay', type=float, default=5e-5, help='weight_decay for node classification training. (default: 1e-3)')

parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs. (default: 500)')
parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
parser.add_argument('--eval_steps', type=int, default=10, help='(default: 10)')
parser.add_argument("--device", type=int, default=0)


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
    # root = 'data/'
    ########################################################################
    transform = T.Compose([
        T.ToUndirected(),
        T.ToDevice(device),
    ])
    data = get_dataset(root, args.dataset, transform=transform)
    train_data, valid_data, test_data = T.RandomLinkSplit(num_val=0.05, num_test=0.1,
                                                        is_undirected=True,
                                                        split_labels=True,
                                                        add_negative_train_samples=False)(data)

    assert args.mask in ['path', 'edge', 'none']
    if args.mask == 'path':
        mask = MaskPath(p=args.p, 
                        num_nodes=data.num_nodes, 
                        start=args.start,
                        walk_length=args.encoder_layers+1)
    elif args.mask == 'edge':
        mask = MaskEdge(p=args.p)
    else:
        mask = NullMask() # vanilla GAE

    encoder = GNNEncoder(in_channels=data.num_features, 
                         hidden_channels=args.encoder_channels, 
                         out_channels=args.encoder_channels,
                         num_layers=args.encoder_layers, 
                         dropout=args.encoder_dropout,
                         norm=args.encoder_norm, 
                         layer=args.layer, 
                         activation=args.encoder_activation)
    
    decoder = EdgeDecoder(in_channels=args.encoder_channels, 
                          hidden_channels=args.decoder_channels,
                          num_layers=args.decoder_layers, 
                          dropout=args.decoder_dropout,
                          norm=args.decoder_norm)
    
    model = lrGAE(encoder, decoder, 
                    left=args.left,
                    right=args.right).to(device)
    print(model)
    print(mask)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)    
    
    for epoch in tqdm(range(1, 1 + args.epochs), 'Link prediction pretraining'):
    
        optimizer.zero_grad()
        model.train()
        remaining_graph, masked_graph = mask(train_data)
        loss = model.train_step(remaining_graph, masked_graph)
        loss.backward()
        if args.grad_norm > 0:
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)        
        optimizer.step()
        
        if epoch % args.eval_steps == 0:
            print('Evaluating...')
            valid_auc, valid_ap = model.eval_linkpred(valid_data)       
            test_auc, test_ap = model.eval_linkpred(test_data)                  
            print(f'Link prediction valid_auc: {valid_auc:.2%}, valid_ap: {valid_ap:.2%}')
            print(f'Link prediction test_auc: {test_auc:.2%}, test_ap: {test_ap:.2%}')

if __name__ == "__main__":
    main()
