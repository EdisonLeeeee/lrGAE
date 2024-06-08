import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
# custom modules
from lrgae.dataset import get_dataset
from lrgae.decoders import CrossCorrelationDecoder, EdgeDecoder, FeatureDecoder
from lrgae.encoders import GNNEncoder
from lrgae.masks import MaskFeature, NullMask
from lrgae.models import GraphMAE2
from lrgae.utils import set_seed, tab_printer
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="Cora",
                    help="Datasets. (default: Cora)")
parser.add_argument('--seed', type=int, default=2024,
                    help='Random seed for model and dataset. (default: 2024)')

parser.add_argument("--layer", nargs="?", default="gcn",
                    help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="prelu",
                    help="Activation function for GNN encoder, (default: prelu)")
parser.add_argument('--encoder_channels', type=int, default=1024,
                    help='Channels of hidden representation. (default: 1024)')
parser.add_argument('--encoder_layers', type=int, default=1,
                    help='Number of layers for encoder. (default: 1)')
parser.add_argument('--encoder_dropout', type=float, default=0.2,
                    help='Dropout probability of encoder. (default: 0.8)')
parser.add_argument("--encoder_norm", nargs="?",
                    default="none", help="Normalization (default: none)")

parser.add_argument('--decoder_channels', type=int, default=32,
                    help='Channels of decoder layers. (default: 32)')
parser.add_argument("--decoder_activation", nargs="?", default="prelu",
                    help="Activation function for GNN encoder, (default: prelu)")
parser.add_argument('--decoder_layers', type=int, default=1,
                    help='Number of layers for decoders. (default: 2)')
parser.add_argument('--decoder_dropout', type=float, default=0.2,
                    help='Dropout probability of decoder. (default: 0.2)')
parser.add_argument("--decoder_norm", nargs="?",
                    default="none", help="Normalization (default: none)")

parser.add_argument('--p', type=float, default=0.7,
                    help='Mask ratio or sample ratio for MaskNode')
parser.add_argument("--remask_rate", type=float, default=0.5)
parser.add_argument("--alpha", type=float, default=3,
                    help="`pow`coefficient for `sce` loss")
parser.add_argument("--remask_method", type=str, default="fixed")
parser.add_argument("--mask_type", type=str,
                    default="mask", help="`mask` or `drop`")
parser.add_argument("--mask_method", type=str, default="random")
parser.add_argument("--replace_rate", type=float, default=0.0)
parser.add_argument("--num_remasking", type=int, default=3)
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate for training. (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='weight_decay for link prediction training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0,
                    help='grad_norm for training. (default: 1.0.)')

parser.add_argument('--epochs', type=int, default=1500,
                    help='Number of training epochs. (default: 1500)')
parser.add_argument('--eval_steps', type=int, default=50, help='(default: 50)')
parser.add_argument("--device", type=int, default=0)


def main():

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    set_seed(args.seed)

    if args.device < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # (!IMPORTANT) Specify the path to your dataset directory ##############
    root = '~/public_data/pyg_data'  # my root directory
    # root = '../data/'
    ########################################################################
    transform = T.Compose([
        T.ToUndirected(),
        T.ToDevice(device),
        # T.NormalizeFeatures(),
    ])
    data = get_dataset(root, args.dataset, transform=transform)
    train_data, valid_data, test_data = T.RandomLinkSplit(num_val=0.05, num_test=0.1,
                                                          is_undirected=True,
                                                          split_labels=True,
                                                          add_negative_train_samples=False)(data)
    encoder = GNNEncoder(in_channels=data.num_features,
                         hidden_channels=args.encoder_channels,
                         out_channels=args.encoder_channels,
                         num_layers=args.encoder_layers,
                         dropout=args.encoder_dropout,
                         norm=args.encoder_norm,
                         layer=args.layer,
                         activation=args.encoder_activation)
    neck = nn.Linear(args.encoder_channels, args.encoder_channels, bias=False)
    decoder = GNNEncoder(in_channels=args.encoder_channels,
                         hidden_channels=args.decoder_channels,
                         out_channels=data.num_features,
                         num_layers=args.decoder_layers,
                         dropout=args.decoder_dropout,
                         norm=args.decoder_norm,
                         layer=args.layer,
                         activation=args.decoder_activation,
                         add_last_act=False,
                         add_last_bn=False)

    model = GraphMAE2(encoder=encoder, decoder=decoder, neck=neck,
                      alpha=args.alpha,
                      num_remasking=args.num_remasking,
                      replace_rate=args.replace_rate,
                      remask_rate=args.remask_rate,
                      remask_method=args.remask_method,
                      mask_rate=args.p,
                      ).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    pbar = tqdm(range(1, 1 + args.epochs))
    best_test_metric = None
    best_valid_metric = None
    for epoch in pbar:
        optimizer.zero_grad()
        model.train()
        loss = model.train_step(train_data)
        loss.backward()
        if args.grad_norm > 0:
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()
        pbar.set_description(f'Loss: {loss.item():.4f}')

        if epoch % args.eval_steps == 0:
            print(f'\nEvaluating on epoch {epoch}...')
            val_results = model.eval_linkpred(valid_data)
            valid_auc, valid_ap = val_results['auc'], val_results['ap']
            test_results = model.eval_linkpred(test_data)
            test_auc, test_ap = test_results['auc'], test_results['ap']
            if best_valid_metric is None or best_valid_metric[0] < valid_auc:
                best_test_metric = test_auc, test_ap
                best_valid_metric = valid_auc, valid_ap
            print(
                f'Link prediction valid_auc: {valid_auc:.2%}, valid_ap: {valid_ap:.2%}')
            print(
                f'Link prediction test_auc: {test_auc:.2%}, test_ap: {test_ap:.2%}')
    print(
        f'Link prediction on {args.dataset} test_auc: {best_test_metric[0]:.2%}, test_ap: {best_test_metric[1]:.2%}')


if __name__ == "__main__":
    main()
