
# Python version -> 3.8.20

# !pip install numpy==1.26.4

# !pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# !pip install dgl -f https://data.dgl.ai/wheels/repo.html
# !pip install ogb



from input_data import load_data
from model import GraphTransformer
from utils import *
import copy
import itertools

import torch
import dgl
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data

# Set DGL backend to PyTorch
import os
os.environ['DGLBACKEND'] = 'pytorch'

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
dgl.seed(seed)
dgl.random.seed(seed)

(
    _, _, g, 
    train_positive_edges, train_negative_edges, 
    valid_positive_edges, valid_negative_edges, 
    test_positive_edges, test_negative_edges
) = load_data()

# Model, optimizer, and loss setup
in_dim = g.ndata['feat'].size(1)
hidden_dim = 80  # Can be tuned
num_epochs = 2000  # Can be tuned
learning_rate = 0.00005  # Can be tuned
num_layers = 6  # Can be tuned
n_heads = 4  # Can be tuned

model = GraphTransformer(in_dim, hidden_dim, n_heads, num_layers)
predictor = LinkPredictor(hidden_dim)
optimizer = torch.optim.Adam(
    itertools.chain(model.parameters(), predictor.parameters()), lr=learning_rate
)
loss_fn = nn.BCEWithLogitsLoss()

# Training loop
best_val_auc = 0
best_model = None

for epoch in range(num_epochs):
    loss, emb = train_link_prediction(
        model, predictor, g, train_positive_edges, train_negative_edges, optimizer, loss_fn
    )
    
    train_auc, train_ap, _, _ = evaluate(
        model, predictor, g, train_positive_edges, train_negative_edges
    )
    val_auc, val_ap, _, _ = evaluate(
        model, predictor, g, valid_positive_edges, valid_negative_edges
    )
    
    print(
        f"Epoch {epoch}: Loss = {loss:.4f}, Train_AUC = {train_auc:.4f}, Train_AP = {train_ap:.4f}, "
        f"Val_AUC = {val_auc:.4f}, Val_AP = {val_ap:.4f}"
    )
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model = copy.deepcopy(model)
        torch.save(best_model.state_dict(), "best_model.pt")

best_model.load_state_dict(torch.load("best_model.pt"))

test_auc, test_ap, lab, log = evaluate(best_model, predictor, g, test_positive_edges, test_negative_edges)
print(f'Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')












