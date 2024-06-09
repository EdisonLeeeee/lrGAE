import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit, TUDataset
from torch_geometric.utils import index_to_mask


def load_dataset(root: str, name: str, transform=None) -> Data:
    if transform is None:
        def transform(x): return x

    if name in {'arxiv', 'products', 'mag'}:
        from ogb.nodeproppred import PygNodePropPredDataset
        print('loading ogb dataset...')
        dataset = PygNodePropPredDataset(root=root, name=f'ogbn-{name}')
        if name in ['mag']:
            rel_data = dataset[0]
            # We are only interested in paper <-> paper relations.
            data = Data(
                x=rel_data.x_dict['paper'],
                edge_index=rel_data.edge_index_dict[(
                    'paper', 'cites', 'paper')],
                y=rel_data.y_dict['paper'])
            data = transform(data)
            split_idx = dataset.get_idx_split()
            data.train_mask = index_to_mask(
                split_idx['train']['paper'], data.num_nodes)
            data.val_mask = index_to_mask(
                split_idx['valid']['paper'], data.num_nodes)
            data.test_mask = index_to_mask(
                split_idx['test']['paper'], data.num_nodes)
        else:
            data = transform(dataset[0])
            split_idx = dataset.get_idx_split()
            data.train_mask = index_to_mask(split_idx['train'], data.num_nodes)
            data.val_mask = index_to_mask(split_idx['valid'], data.num_nodes)
            data.test_mask = index_to_mask(split_idx['test'], data.num_nodes)

    elif name in {'Cora', 'Citeseer', 'Pubmed'}:
        dataset = Planetoid(root, name)
        data = transform(dataset[0])

    elif name == 'Reddit':
        dataset = Reddit(osp.join(root, name))
        data = transform(dataset[0])
    elif name in {'Photo', 'Computers'}:
        dataset = Amazon(root, name)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    elif name in {'CS', 'Physics'}:
        dataset = Coauthor(root, name)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    elif name in ['NCI1', 'DD', 'PROTEINS', 'COLLAB',
                  'MUTAG', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']:
        data = TUDataset(root, name, transform)
    else:
        raise ValueError(name)
    return data
