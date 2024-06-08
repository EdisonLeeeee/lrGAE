from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

# custom modules
from lrgae.decoders import (CrossCorrelationDecoder, DotProductEdgeDecoder,
                            EdgeDecoder)


class NodeClasEvaluator:
    def __init__(self,
                 lr: float = 0.01,
                 weight_decay: float = 0.,
                 batch_size: int = 512,
                 mode: str = 'cat',
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

        with torch.no_grad():
            self.eval()
            embedding = model.encode(data.x.to(self.device),
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
            for epoch in range(1, self.epochs + 1):
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
        z = model.encode(data.x.to(self.device),
                         data.edge_index.to(self.device))

        decoder = model.decoder
        if not isinstance(decoder, (EdgeDecoder, CrossCorrelationDecoder)):
            decoder = DotProductEdgeDecoder()

        if isinstance(left, (list, tuple)):
            left = [z[l] for l in self.left]
        else:
            left = z[self.left]
        if isinstance(right, (list, tuple)):
            right = [z[r] for r in self.right]
        else:
            right = z[self.right]

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
