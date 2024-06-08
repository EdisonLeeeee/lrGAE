import argparse

import torch
import torch.nn as nn
import torch_geometric.transforms as T
from tqdm.auto import tqdm

# custom modules
from lrgae.dataset import get_dataset
from lrgae.decoders import EdgeDecoder, FeatureDecoder
from lrgae.encoders import GNNEncoder
from lrgae.masks import MaskEdge, MaskPath, NullMask
from lrgae.models import MaskGAE
from lrgae.utils import set_seed
from lrgae.evaluators import LinkPredEvaluator

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="Cora",
                    help="Datasets. (default: Cora)")
parser.add_argument("--mask", default="path",
                    help="Masking stractegy, `path`, `edge` or `None` (default: path)")
parser.add_argument('--seed', type=int, default=2024,
                    help='Random seed for model and dataset. (default: 2024)')

parser.add_argument("--layer", default="gcn",
                    help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", default="elu",
                    help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=128,
                    help='Channels of hidden representation. (default: 128)')
parser.add_argument('--encoder_layers', type=int, default=1,
                    help='Number of layers for encoder. (default: 1)')
parser.add_argument('--encoder_dropout', type=float, default=0.8,
                    help='Dropout probability of encoder. (default: 0.8)')
parser.add_argument("--encoder_norm",
                    default="batchnorm", help="Normalization (default: none)")

parser.add_argument('--decoder_channels', type=int, default=128,
                    help='Channels of decoder layers. (default: 128)')
parser.add_argument('--decoder_layers', type=int, default=2,
                    help='Number of layers for decoders. (default: 2)')
parser.add_argument('--decoder_dropout', type=float, default=0.2,
                    help='Dropout probability of decoder. (default: 0.2)')
parser.add_argument("--decoder_norm",
                    default="none", help="Normalization (default: none)")

parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate for training. (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='weight_decay for link prediction training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0,
                    help='grad_norm for training. (default: 1.0.)')

parser.add_argument('--alpha', type=float, default=0.003,
                    help='loss weight for degree prediction. (default: 0.001)')
parser.add_argument("--start", default="node",
                    help="Which Type to sample starting nodes for random walks, (default: node)")
parser.add_argument('--p', type=float, default=0.7,
                    help='Mask ratio or sample ratio for MaskEdge/MaskPath')

parser.add_argument('--epochs', type=int, default=500,
                    help='Number of training epochs. (default: 500)')
parser.add_argument('--eval_steps', type=int, default=50, help='(default: 50)')
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
data = get_dataset(root, args.dataset, transform=transform)
evaluator = LinkPredEvaluator(device=device)
train_data, valid_data, test_data = T.RandomLinkSplit(num_val=0.05, num_test=0.1,
                                                      is_undirected=True,
                                                      split_labels=True,
                                                      add_negative_train_samples=False)(data)

assert args.mask in ['path', 'edge', 'none']
if args.mask == 'path':
    mask = MaskPath(p=args.p,
                    num_nodes=data.num_nodes,
                    start=args.start,
                    walk_length=args.encoder_layers + 1)
elif args.mask == 'edge':
    mask = MaskEdge(p=args.p)
else:
    mask = NullMask()  # vanilla GAE

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
degree_decoder = FeatureDecoder(in_channels=args.encoder_channels,
                                hidden_channels=args.decoder_channels,
                                num_layers=args.decoder_layers,
                                dropout=args.decoder_dropout,
                                norm=args.decoder_norm)

model = MaskGAE(encoder, decoder, mask,
                degree_decoder=degree_decoder).to(device)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)
pbar = tqdm(range(1, 1 + args.epochs))
best_test_metric = None
best_valid_metric = None
for epoch in pbar:
    optimizer.zero_grad()
    model.train()
    loss = model.train_step(train_data, alpha=args.alpha)
    loss.backward()
    if args.grad_norm > 0:
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
    optimizer.step()
    pbar.set_description(f'Loss: {loss.item():.4f}')

    if epoch % args.eval_steps == 0:
        print(f'\nEvaluating on epoch {epoch}...')
        val_results = evaluator.evaluate(model, valid_data)
        valid_auc, valid_ap = val_results['auc'], val_results['ap']
        test_results = evaluator.evaluate(model, test_data)
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
