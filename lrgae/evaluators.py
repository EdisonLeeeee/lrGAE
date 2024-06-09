from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (average_precision_score, 
                             roc_auc_score,
                             normalized_mutual_info_score,
                             adjusted_rand_score)

# custom modules
from lrgae.decoders import (CrossCorrelationDecoder, 
                            DotProductEdgeDecoder,
                            EdgeDecoder)
from lrgae.kmeans import kmeans


class NodeClasEvaluator:
    def __init__(self,
                 lr: float = 0.01,
                 weight_decay: float = 0.,
                 batch_size: int = 512,
                 mode: str = 'last',
                 l2_normalize: bool = False,
                 runs: int = 1,
                 epochs: int = 100,
                 device: str = 'cpu',
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.mode = mode
        self.l2_normalize = l2_normalize
        self.runs = runs
        self.epochs = epochs
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

        y = data.y.squeeze().to(self.device)
        train_x, train_y = embedding[data.train_mask], y[data.train_mask]
        val_x, val_y = embedding[data.val_mask], y[data.val_mask]
        test_x, test_y = embedding[data.test_mask], y[data.test_mask]
        results = self.linear_probing(train_x,
                                      train_y,
                                      val_x,
                                      val_y,
                                      test_x,
                                      test_y)

        return {'acc': np.mean(results)}

    def linear_probing(self,
                       train_x: Tensor,
                       train_y: Tensor,
                       val_x: Tensor,
                       val_y: Tensor,
                       test_x: Tensor,
                       test_y: Tensor,
                       ):

        @torch.no_grad()
        def test(loader):
            classifier.eval()
            logits = []
            labels = []
            for x, y in loader:
                logits.append(classifier(x))
                labels.append(y)
            logits = torch.cat(logits, dim=0).cpu()
            labels = torch.cat(labels, dim=0).cpu()
            logits = logits.argmax(1)
            return (logits == labels).float().mean().item()

        num_classes = train_y.max().item() + 1
        classifier = nn.Linear(train_x.size(1), num_classes).to(self.device)

        train_loader = DataLoader(TensorDataset(train_x, train_y),
                                  batch_size=self.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(TensorDataset(val_x, val_y),
                                batch_size=20000)
        test_loader = DataLoader(TensorDataset(test_x, test_y),
                                 batch_size=20000)

        results = []
        for _ in range(self.runs):
            nn.init.xavier_uniform_(classifier.weight.data)
            nn.init.zeros_(classifier.bias.data)
            optimizer = torch.optim.Adam(classifier.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)

            best_val_metric = test_metric = 0
            for _ in range(1, self.epochs + 1):
                classifier.train()
                for x, y in train_loader:
                    optimizer.zero_grad()
                    F.cross_entropy(classifier(x), y).backward()
                    optimizer.step()

                val_metric, test_metric = test(val_loader), test(test_loader)
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_test_metric = test_metric
            results.append(best_test_metric)

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
        
# def linear_probing_cv(x, y, test_ratio=0.1):
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.metrics import accuracy_score, f1_score
#     from sklearn.model_selection import GridSearchCV, train_test_split
#     from sklearn.multiclass import OneVsRestClassifier
#     from sklearn.preprocessing import OneHotEncoder, normalize
#     from sklearn.svm import SVC

#     def prob_to_one_hot(y_pred):
#         ret = np.zeros(y_pred.shape, bool)
#         indices = np.argmax(y_pred, axis=1)
#         for i in range(y_pred.shape[0]):
#             ret[i][indices[i]] = True
#         return ret

#     x = x.cpu().numpy()
#     y = y.cpu().numpy().reshape(-1, 1)

#     onehot_encoder = OneHotEncoder(categories='auto').fit(y)
#     y = onehot_encoder.transform(y).toarray().astype(bool)

#     X_train, X_test, y_train, y_test = train_test_split(
#         x, y, test_size=test_ratio)
#     logreg = LogisticRegression(solver='liblinear')
#     c = 2.0 ** np.arange(-10, 10)

#     clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
#                        param_grid=dict(estimator__C=c), n_jobs=4, cv=5,
#                        verbose=0)

#     clf.fit(X_train, y_train)

#     y_pred = clf.predict_proba(X_test)
#     y_pred = prob_to_one_hot(y_pred)

#     micro = f1_score(y_test, y_pred, average="micro")
#     print(micro)
#     return [micro]
