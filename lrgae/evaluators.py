from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Data, HeteroData
from sklearn.metrics import (average_precision_score,
                             roc_auc_score,
                             f1_score,
                             normalized_mutual_info_score,
                             adjusted_rand_score,
                             accuracy_score)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
# custom modules
from lrgae.decoders import (CrossCorrelationDecoder,
                            DotProductEdgeDecoder,
                            EdgeDecoder)
from lrgae.kmeans import kmeans


class NodeClasEvaluator:
    def __init__(self,
                 lr: float = 0.01,
                 weight_decay: float = 0.,
                 batch_size: int = 2048,
                 mode: str = 'last',
                 l2_normalize: bool = False,
                 runs: int = 1,
                 epochs: int = 100,
                 device: str = 'cpu',
                 node_type: str = None,
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.mode = mode
        self.l2_normalize = l2_normalize
        self.runs = runs
        self.epochs = epochs
        self.device = device
        self.node_type = node_type # hetero graph only

    def evaluate(self, model, data):
        model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            if isinstance(data, Data):
                embeddings = model(data.x, data.edge_index)[1:]
            else:
                embedding_dict = model(data.x_dict, data.edge_index_dict)[1:]
                embeddings = [e[self.node_type] for e in embedding_dict]
                                   
            if self.mode == 'cat':
                embeddings = torch.cat(embeddings, dim=-1)
            else:
                embeddings = embeddings[-1]

            if self.l2_normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

        if isinstance(data, HeteroData):
            data = data[self.node_type]
        y = data.y.squeeze().to(self.device)

        multi_class = y.dim() > 1
            
        train_x, train_y = embeddings[data.train_mask], y[data.train_mask]
        val_x, val_y = embeddings[data.val_mask], y[data.val_mask]
        test_x, test_y = embeddings[data.test_mask], y[data.test_mask]

        num_features = embeddings.size(1)
        if multi_class:
            num_classes = y.size(1)
        else:
            num_classes = y.max().item() + 1
        LR = LogisticRegression(num_features, num_classes,
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                batch_size=self.batch_size,
                                epochs=self.epochs,
                                multi_class=multi_class,
                                device=self.device)

        results = []
        for _ in range(self.runs):
            LR.reset_parameters()
            results.append(LR.fit(train_x,
                                  train_y,
                                  test_x,
                                  test_y,
                                  val_x,
                                  val_y))
        if multi_class:
            return {'micro-f1': np.mean(results)}
        else:
            return {'acc': np.mean(results)}


class GraphClasEvaluator:
    def __init__(self,
                 pooling: str = 'mean',
                 lr: float = 0.01,
                 weight_decay: float = 0.,
                 batch_size: int = 2048,
                 mode: str = 'last',
                 l2_normalize: bool = False,
                 runs: int = 5,
                 epochs: int = 100,
                 classifier: str = 'svm',
                 device: str = 'cpu',
                 ):
        assert runs > 1
        assert classifier in ['lr', 'svm']
        if pooling == 'mean':
            self.pooling = global_mean_pool
        elif pooling == 'sum':
            self.pooling = global_add_pool
        elif pooling == 'max':
            self.pooling = global_max_pool
        else:
            raise ValueError(pooling)
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.mode = mode
        self.l2_normalize = l2_normalize
        self.runs = runs
        self.epochs = epochs
        self.classifier = classifier
        self.device = device

    def evaluate(self, model, loader):
        model.eval()
        y = []
        embeddings = []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                embedding = model(data.x, data.edge_index)[1:]
                embedding = [self.pooling(x, data.batch) for x in embedding]
                if self.mode == 'cat':
                    embedding = torch.cat(embedding, dim=-1)
                else:
                    embedding = embedding[-1]
                if self.l2_normalize:
                    embedding = F.normalize(embedding, p=2, dim=1)
                embeddings.append(embedding.cpu())
                y.append(data.y.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        y = torch.cat(y).squeeze()
        if self.classifier == 'svm':
            results = evaluate_graph_embeddings_using_svm(embeddings.cpu(), y.cpu())
        else:
            num_features = embeddings.size(1)
            num_classes = y.max().item() + 1
            LR = LogisticRegression(num_features, num_classes,
                                    lr=self.lr,
                                    weight_decay=self.weight_decay,
                                    batch_size=self.batch_size,
                                    epochs=self.epochs,
                                    device=self.device)
            
            kf = StratifiedKFold(n_splits=self.runs, shuffle=True, random_state=0)
            results = []
            for train_index, test_index in kf.split(embeddings, y):
                train_x = embeddings[train_index].to(self.device)
                train_y = y[train_index].to(self.device)
    
                test_x = embeddings[test_index].to(self.device)
                test_y = y[test_index].to(self.device)
                LR.reset_parameters()
                results.append(LR.fit(train_x,
                                      train_y,
                                      test_x,
                                      test_y))

        return {'acc': np.mean(results)}

def evaluate_graph_embeddings_using_svm(embeddings, labels):

    
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    results = []
    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        acc = accuracy_score(y_test, preds)
        results.append(acc)
    return results
    
class LinkPredEvaluator:
    def __init__(self,
                 batch_size: int = 2**16,
                 device: str = 'cpu',
                 ):
        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def batch_predict(self, decoder, left, right, edge_index):
        preds = []
        for edge in DataLoader(TensorDataset(edge_index.t()),
                               batch_size=self.batch_size):
            edge = edge[0].t()
            preds.append(decoder(left, right, edge).squeeze())
        pred = torch.cat(preds, dim=0)
        return pred.cpu()

    def evaluate(self, model, data,
                 left: Union[int, tuple, list] = -1,
                 right: Union[int, tuple, list] = -1):

        model.eval()
        z = model(data.x.to(self.device),
                  data.edge_index.to(self.device))

        decoder = model.decoder
        if not isinstance(decoder, (EdgeDecoder, CrossCorrelationDecoder)):
            decoder = DotProductEdgeDecoder()

        if isinstance(left, (list, tuple)):
            left = [z[l] for l in left]
        else:
            left = z[left]
        if isinstance(right, (list, tuple)):
            right = [z[r] for r in right]
        else:
            right = z[right]

        pos_pred = self.batch_predict(decoder, left, right,
                                      data.pos_edge_label_index)
        neg_pred = self.batch_predict(decoder, left, right,
                                      data.neg_edge_label_index)

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))

        y = torch.cat([pos_y, neg_y], dim=0)

        return {'auc': roc_auc_score(y, pred),
                'ap': average_precision_score(y, pred)}


class GraphClusterEvaluator:
    def __init__(self,
                 mode: str = 'last',
                 l2_normalize: bool = False,
                 runs: int = 1,
                 device: str = 'cpu',
                 ):
        self.mode = mode
        self.l2_normalize = l2_normalize
        self.runs = runs
        self.device = device

    def evaluate(self, model, data):
        model.eval()

        with torch.no_grad():
            embedding = model(data.x.to(self.device),
                              data.edge_index.to(self.device))[1:]
            if self.mode == 'cat':
                embedding = torch.cat(embedding, dim=-1)
            else:
                embedding = embedding[-1]

            if self.l2_normalize:
                embedding = F.normalize(embedding, p=2, dim=1)

        y = data.y.squeeze()

        num_clusters = y.max().item() + 1
        nmis = []
        aris = []
        for _ in range(self.runs):
            clusters, _ = kmeans(embedding, num_clusters)
            nmi = normalized_mutual_info_score(y.cpu(), clusters.cpu())
            ari = adjusted_rand_score(y.cpu(), clusters.cpu())
            nmis.append(nmi)
            aris.append(ari)

        return {'NMI': np.mean(nmis), 'ARI': np.mean(aris)}


class LogisticRegression:
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 lr: float = 0.01,
                 weight_decay: float = 0.,
                 batch_size: int = 512,
                 epochs: int = 100,
                 multi_class: bool = False,
                 device: str = 'cpu',
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.multi_class = multi_class
        self.device = device
        self.classifier = nn.Linear(in_channels, out_channels).to(device)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.classifier.weight.data)
        nn.init.zeros_(self.classifier.bias.data)

    def fit(self, train_x, train_y, test_x, test_y, val_x=None, val_y=None):
        classifier = self.classifier
        optimizer = torch.optim.Adam(classifier.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        train_loader = DataLoader(TensorDataset(train_x, train_y),
                                  batch_size=self.batch_size,
                                  shuffle=True)
        if val_x is not None and val_y is not None:
            val_loader = DataLoader(TensorDataset(val_x, val_y),
                                    batch_size=20000)
        else:
            val_loader = None
        test_loader = DataLoader(TensorDataset(test_x, test_y),
                                 batch_size=20000)

        best_val_metric = test_metric = 0
        if self.multi_class:
            loss_fn = F.binary_cross_entropy_with_logits
        else:
            loss_fn = F.cross_entropy
        for _ in range(1, self.epochs + 1):
            classifier.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                loss_fn(classifier(x), y).backward()
                optimizer.step()

            test_metric = self.evaluate(test_loader)
            if val_loader is not None:
                val_metric = self.evaluate(val_loader)
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_test_metric = test_metric
            else:
                best_test_metric = test_metric

        return best_test_metric

    @torch.no_grad()
    def evaluate(self, loader):
        classifier = self.classifier
        classifier.eval()
        logits = []
        labels = []
        for x, y in loader:
            logits.append(classifier(x))
            labels.append(y)
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        if self.multi_class:
            logits = (logits > 0).float()
            return f1_score(logits, labels, average='micro')
        else:
            # acc
            logits = logits.argmax(1)
            return accuracy_score(logits, labels)
