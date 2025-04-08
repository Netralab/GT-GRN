import torch
import torch.nn as nn
import dgl
import dgl.sparse as dglsp
import dgl.function as fn
import random
import itertools
from dgl.data import CoraGraphDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc


import torch


class LinkPredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        x = torch.cat([src, dst], dim=1)
        x = self.decoder(x)
        return x

def train_link_prediction(model, predictor, graph, pos_edges, neg_edges, optimizer, loss_fn):
    model.train()
    predictor.train()

    optimizer.zero_grad()


    h = model(graph, graph.ndata['feat'], graph.ndata['PE'], graph.ndata['global'])
    pos_logits = predictor(h, pos_edges)
    neg_logits = predictor(h, neg_edges)
    pos_labels = torch.ones_like(pos_logits)
    neg_labels = torch.zeros_like(neg_logits)

    loss = loss_fn(pos_logits, pos_labels) + loss_fn(neg_logits, neg_labels)
    
    loss.backward()
    optimizer.step()
    return loss.item(), h

def evaluate(model, predictor, graph, pos_edges, neg_edges):
    model.eval()
    predictor.eval()

    with torch.no_grad():
        h = model(graph, graph.ndata['feat'], graph.ndata['PE'], graph.ndata['global'])
        pos_logits = predictor(h, pos_edges)
        neg_logits = predictor(h, neg_edges)

    logits = torch.cat([pos_logits, neg_logits])

    

    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
    return roc_auc_score(labels.cpu(), logits.cpu()), average_precision_score(labels.cpu(), logits.cpu()), labels.cpu(), logits.cpu()