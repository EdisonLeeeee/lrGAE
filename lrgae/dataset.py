import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit, TUDataset, HGBDataset
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import index_to_mask, degree, one_hot

def generate_random_splits(num_nodes, train_ratio, val_ratio):
    test_ratio = 1 - train_ratio - val_ratio
    train_mask = torch.full((num_nodes, ), False, dtype=torch.bool)
    val_mask = torch.full((num_nodes, ), False, dtype=torch.bool)
    test_mask = torch.full((num_nodes, ), False, dtype=torch.bool)

    permute = torch.randperm(num_nodes)
    train_idx = permute[: int(train_ratio * num_nodes)]
    val_idx = permute[int(train_ratio * num_nodes)
                          : int((train_ratio + val_ratio) * num_nodes)]
    test_idx = permute[int(1 - test_ratio * num_nodes):]
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask
    
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



class OneHotDegree(T.BaseTransform):
    def __init__(
        self,
        max_degree: int,
        in_degree: bool = False,
        cat: bool = True,
    ) -> None:
        self.max_degree = max_degree
        self.in_degree = in_degree
        self.cat = cat

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = degree(idx, data.num_nodes, dtype=torch.long)
        deg[deg > self.max_degree] = self.max_degree
        deg = one_hot(deg, num_classes=self.max_degree + 1)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.max_degree})'


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
        dataset = TUDataset(root, name, transform, use_node_attr=True)
        max_degree = 0.
        for data in dataset:
            max_degree = max(max_degree, degree(data.edge_index[0], 
                                                dtype=torch.long).max().item())
        max_degree = min(400, max_degree)
        add_transform_to_dataset(dataset, OneHotDegree(max_degree))
        return dataset
    elif name in ['ACM', 'DBLP', 'IMDB', 'FreeBase']:
        data = HGBDataset(root=f'{root}/HGBDataset', name=name.lower(), transform=transform)[0]
        if name == 'ACM':
            data['term'].x = torch.zeros(data['term'].num_nodes, 1)
        else:
            for nt in data.node_types:
                if data[nt].get('x') is None:
                    data[nt].x = torch.eye(data[nt].num_nodes)        
                    
        node_type = [t for t in data.node_types if data[t].get('y') is not None][0]
        train_mask, val_mask, test_mask = generate_random_splits(
            num_nodes=data[node_type].num_nodes,
            train_ratio=0.6,
            val_ratio=0.2,
        )
        data[node_type].train_mask = train_mask
        data[node_type].val_mask = val_mask
        data[node_type].test_mask = test_mask        
        # train_mask = data[node_type].train_mask
        # val_mask = data[node_type].get('val_mask')
        # if val_mask is None:
        #     train_idx = train_mask.nonzero().view(-1)
        #     num_valid_samples = int(train_idx.size(0)*0.2)
        #     valid_idx = train_idx[torch.randperm(train_idx.size(0))[
        #         :num_valid_samples]]
        #     train_mask[valid_idx] = False
        #     val_mask = index_to_mask(valid_idx, data[node_type].num_nodes) 
        # data[node_type].val_mask = val_mask
    else:
        raise ValueError(name)
    return data
