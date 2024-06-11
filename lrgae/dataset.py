import os.path as osp
import torch

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit, TUDataset
from torch_geometric.utils import index_to_mask, degree

class NullTransform(T.BaseTransform):
    def forward(self, data):
        return data

class OneHotLabel(T.BaseTransform):
    def __init__(self, num_classes):
        self.num_classes = num_classes
    def forward(self, data):
        x = data.x
        y = F.one_hot(data.y.view(-1), 
                      num_classes=self.num_classes)
        if x is not None:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, y.to(x.dtype)], dim=-1)
        else:
            data.x = y.to(torch.float)        
        return data

def add_transform_to_dataset(dataset, transform):
    trans = dataset.transform
    if isinstance(trans, T.Compose):
        trans.transforms.append(transform)
    elif isinstance(trans, T.BaseTransform):
        dataset.transform = T.Compose([trans, transform])
    else:
        dataset.transform = transform
        
def load_dataset(root: str, name: str, transform=None) -> Data:
    if transform is None:
        transform = NullTransform()

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
    elif name in ['IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'COLLAB', 
                  'MUTAG', 'REDDIT-BINARY','NCI1', 'REDDIT-MULTI-5K', 'DD']:
        dataset = TUDataset(root, name, transform)
        max_degree = 0.
        for data in dataset:
            max_degree = max(max_degree, degree(data.edge_index[0], 
                                                dtype=torch.long).max().item())
        add_transform_to_dataset(dataset, T.OneHotDegree(max_degree))
        return dataset
    else:
        raise ValueError(name)
    return data
