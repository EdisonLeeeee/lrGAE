import argparse
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader


# custom modules
from lrgae.dataset import load_dataset
from lrgae.decoders import EdgeDecoder
from lrgae.encoders import GNNEncoder
from lrgae.models import GAE
from lrgae.utils import set_seed
from lrgae.evaluators import GraphClasEvaluator

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="MUTAG",
                    help="Datasets. (default: MUTAG)")
parser.add_argument('--seed', type=int, default=2024,
                    help='Random seed for model and dataset. (default: 2024)')

parser.add_argument("--layer", default="gcn",
                    help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", default="relu",
                    help="Activation function for GNN encoder, (default: relu)")
parser.add_argument('--encoder_channels', type=int, default=64,
                    help='Channels of hidden representation. (default: 64)')
parser.add_argument('--encoder_layers', type=int, default=2,
                    help='Number of layers for encoder. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.5,
                    help='Dropout probability of encoder. (default: 0.5)')
parser.add_argument("--encoder_norm",
                    default="none", help="Normalization (default: none)")
parser.add_argument("--pooling", default="sum",
                    help="Pooling layer, (default: sum)")

parser.add_argument('--decoder_channels', type=int, default=32,
                    help='Channels of decoder layers. (default: 32)')
parser.add_argument('--decoder_layers', type=int, default=2,
                    help='Number of layers for decoders. (default: 2)')
parser.add_argument('--decoder_dropout', type=float, default=0.2,
                    help='Dropout probability of decoder. (default: 0.2)')
parser.add_argument("--decoder_norm",
                    default="none", help="Normalization (default: none)")

parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate for training. (default: 0.01)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Learning batch size. (default: 32)')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay for link prediction training. (default: 0.)')
parser.add_argument('--grad_norm', type=float, default=1.0,
                    help='grad_norm for training. (default: 1.0.)')

parser.add_argument('--l2_normalize', action='store_true',
                    help='Whether to use l2 normalize output embedding. (default: False)')
parser.add_argument('--graphclas_lr', type=float, default=0.01,
                    help='Learning rate for graph classification linear probing. (default: 0.01)')
parser.add_argument('--graphclas_weight_decay', type=float, default=5e-5,
                    help='weight_decay for graph classification linear probing. (default: 5e-5)')
parser.add_argument("--mode", default="last",
                    help="Embedding mode `last` or `cat` (default: last)")

parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs. (default: 100)')
parser.add_argument('--runs', type=int, default=10,
                    help='Number of runs. (default: 10)')
parser.add_argument('--eval_steps', type=int, default=5, help='(default: 5)')
parser.add_argument("--device", type=int, default=0)


args = parser.parse_args()

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
])
dataset = load_dataset(root, args.dataset, transform=transform)

evaluator = GraphClasEvaluator(pooling=args.pooling,
                               lr=args.graphclas_lr,
                               weight_decay=args.graphclas_weight_decay,
                               mode=args.mode,
                               l2_normalize=args.l2_normalize,
                               device=device)
encoder = GNNEncoder(in_channels=dataset.num_features,
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

model = GAE(encoder, decoder).to(device)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

best_metric = None
optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)
pbar = tqdm(range(1, 1 + args.epochs))
for epoch in pbar:
    model.train()
    loss_total = 0.
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        loss = model.train_step(data)
        loss.backward()
        if args.grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()
        loss_total += loss.item()
    pbar.set_description(f'Loss: {loss_total:.4f}')

    if epoch % args.eval_steps == 0:
        print(f'Evaluating on epoch {epoch}...')
        results = evaluator.evaluate(model, loader)

        if best_metric is None:
            best_metric = results
        for metric, value in results.items():
            print(f'- Averaged {metric}: {value:.2%}')
            if best_metric[metric] < value:
                best_metric = results

for metric, value in best_metric.items():
    print(f'Best averaged {metric} on {args.dataset}: {value:.2%}')
