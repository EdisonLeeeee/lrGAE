import argparse
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch_geometric.transforms as T
# custom modules
from lrgae.dataset import get_dataset
from lrgae.encoders import GNNEncoder
from lrgae.models import GraphMAE
from lrgae.utils import set_seed
from lrgae.evaluators import NodeClasEvaluator

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="Cora",
                    help="Datasets. (default: Cora)")
parser.add_argument('--seed', type=int, default=2024,
                    help='Random seed for model and dataset. (default: 2024)')

parser.add_argument("--layer", default="gat", help="GNN layer, (default: gat)")
parser.add_argument("--encoder_activation", default="prelu",
                    help="Activation function for GNN encoder, (default: prelu)")
parser.add_argument('--encoder_channels', type=int, default=1024,
                    help='Channels of hidden representation. (default: 1024)')
parser.add_argument('--encoder_layers', type=int, default=2,
                    help='Number of layers for encoder. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.2,
                    help='Dropout probability of encoder. (default: 0.2)')
parser.add_argument("--encoder_norm", default="none",
                    help="Normalization (default: none)")
parser.add_argument("--num_heads", type=int, default=4,
                    help="Number of attention heads for GAT encoders (default: 4)")

parser.add_argument('--decoder_channels', type=int, default=32,
                    help='Channels of decoder layers. (default: 32)')
parser.add_argument("--decoder_activation", default="prelu",
                    help="Activation function for GNN encoder, (default: prelu)")
parser.add_argument('--decoder_layers', type=int, default=1,
                    help='Number of layers for decoders. (default: 1)')
parser.add_argument('--decoder_dropout', type=float, default=0.2,
                    help='Dropout probability of decoder. (default: 0.2)')
parser.add_argument("--decoder_norm", default="none",
                    help="Normalization (default: none)")

parser.add_argument('--p', type=float, default=0.7,
                    help='Mask ratio or sample ratio for MaskNode')
parser.add_argument("--replace_rate", type=float, default=0.0)
parser.add_argument("--alpha", type=float, default=3,
                    help="`pow`coefficient for `sce` loss")

parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate for training. (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='weight_decay for link prediction training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0,
                    help='grad_norm for training. (default: 1.0.)')
parser.add_argument("--mode", default="last",
                    help="Embedding mode `last` or `cat` (default: none)")

parser.add_argument('--l2_normalize', action='store_true',
                    help='Whether to use l2 normalize output embedding. (default: False)')
parser.add_argument('--nodeclas_lr', type=float, default=0.01,
                    help='Learning rate for training. (default: 0.01)')
parser.add_argument('--nodeclas_weight_decay', type=float, default=5e-5,
                    help='weight_decay for node classification training. (default: 5e-5)')

parser.add_argument('--epochs', type=int, default=1500,
                    help='Number of training epochs. (default: 1500)')
parser.add_argument('--runs', type=int, default=1,
                    help='Number of runs. (default: 1)')
parser.add_argument('--eval_steps', type=int, default=50, help='(default: 50)')
parser.add_argument("--device", type=int, default=0)


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
evaluator = NodeClasEvaluator(lr=args.nodeclas_lr,
                              weight_decay=args.nodeclas_weight_decay,
                              mode=args.mode,
                              l2_normalize=args.l2_normalize,
                              runs=args.runs,
                              epochs=args.epochs,
                              device=device)

num_heads = args.num_heads
encoder = GNNEncoder(in_channels=data.num_features,
                     hidden_channels=args.encoder_channels // num_heads,
                     out_channels=args.encoder_channels,
                     num_layers=args.encoder_layers,
                     dropout=args.encoder_dropout,
                     norm=args.encoder_norm,
                     layer=args.layer,
                     num_heads=num_heads,
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

model = GraphMAE(encoder=encoder, decoder=decoder, neck=neck,
                 alpha=args.alpha,
                 replace_rate=args.replace_rate,
                 mask_rate=args.p).to(device)

best_metric = None
optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)
pbar = tqdm(range(1, 1 + args.epochs))
for epoch in pbar:

    optimizer.zero_grad()
    model.train()
    loss = model.train_step(data)
    loss.backward()
    if args.grad_norm > 0:
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
    optimizer.step()
    pbar.set_description(f'Loss: {loss.item():.4f}')

    if epoch % args.eval_steps == 0:
        print(f'\nEvaluating on epoch {epoch}...')
        results = evaluator.evaluate(model, data)

        if best_metric is None:
            best_metric = results
        for metric, value in results.items():
            print(f'- Averaged {metric}: {value:.2%}')
            if best_metric[metric] < value:
                best_metric = results

for metric, value in best_metric.items():
    print(f'Best averaged {metric} on {args.dataset}: {value:.2%}')
