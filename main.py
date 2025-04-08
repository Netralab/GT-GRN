# from input_data import load_data
# from model import GraphTransformer
# from utils import *
# import copy

# import torch
# import dgl
# import dgl.nn as dglnn
# import dgl.sparse as dglsp
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# from torch_geometric.data import Data

# # set dgl backend to pytorch
# import os
# os.environ['DGLBACKEND'] = 'pytorch'

# seed = 42

# # random.seed(seed)
# # torch.manual_seed(seed)
# # np.random.seed(seed)



# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# dgl.seed(seed)
# dgl.random.seed(seed)

# # torch.cuda.manual_seed_all(seed)  # If using GPU
# # torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
# # torch.backends.cudnn.benchmark = False  # Disables automatic optimization for speed



# fin_pos, fin_neg, g, train_positive_edges, train_negative_edges, valid_positive_edges, valid_negative_edges, test_positive_edges, test_negative_edges = load_data()



# # --- Model, Optimizer, and Loss Setup ---
# in_dim = g.ndata['feat'].size(1)
# hidden_dim = 80 # 128, 256, 512

# num_epochs = 2000 # 100, 200, 300 # 1200

# learning= 0.00005 #05 # 0.001, 0.003, 0.0005, 0.00005 #################0.0001
# num_layers = 6 # 4,6,8                  
# n_heads = 4 # 2, 4, 8

# # random.seed(seed)
# # np.random.seed(seed)
# # torch.manual_seed(seed)
# # if device.type == 'cuda':
# #     torch.cuda.manual_seed(seed)
    




# model = GraphTransformer(in_dim, hidden_dim, n_heads, num_layers)
# predictor = LinkPredictor(hidden_dim)
# optimizer = torch.optim.Adam(itertools.chain(model.parameters(), predictor.parameters()), lr=learning)
# loss_fn = nn.BCEWithLogitsLoss()



# # # --- Training Loop ---

# best_val_acc = 0
# best_model = None

# for epoch in range(num_epochs):
#     loss, emb = train_link_prediction(model, predictor, g, fin_pos, fin_neg, optimizer, loss_fn)
#     #loss = train_link_prediction(model, predictor, g, train_positive_edges, train_negative_edges, optimizer, loss_fn)
#     # if epoch % 10 == 0:
#     #     auc = evaluate(model, predictor, g, pos_edges, neg_edges)
#     #     print(f'Epoch {epoch}: Loss = {loss:.4f}, AUC = {auc:.4f}')
#     # if epoch % epoch == 0:
    
#     train_auc, train_ap, tr_lab , tr_log = evaluate(model, predictor, g, fin_pos, fin_neg)
#     # train_auc, train_ap, _, _ = evaluate(model, predictor, g, train_positive_edges, train_negative_edges)
#     val_auc, val_ap, _, _ = evaluate(model, predictor, g, valid_positive_edges, valid_negative_edges)
#     print(f'Epoch {epoch}: Loss = {loss:.4f}, Train_AUC = {train_auc:.4f}, Train_AP = {train_ap:.4f}, Val_AUC = {val_auc:.4f}, Val_AP = {val_ap:.4f}')

#     if train_auc > best_val_acc:
#         best_val_acc = train_auc
#         eph = epoch
#         best_model = copy.deepcopy(model)
#         d = Data(emb=emb, tr_lab=tr_lab, tr_log=tr_log, fin_pos=fin_pos, fin_neg=fin_neg, g=g, eph=eph)
#         torch.save(d, "best_network.pt")


# # --- Final Evaluation ---
# print(eph)
# test_auc, test_ap, lab, log = evaluate(best_model, predictor, g, test_positive_edges, test_negative_edges)
# print(f'Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')


# prec, rec, thr = precision_recall_curve(lab, log)
# aupr_test_new = auc(rec, prec)
# fpr, tpr, thr = roc_curve(lab, log)
# auc_test_new = auc(fpr, tpr)


# print(f'{auc_test_new:.4f}')
# print(f'{aupr_test_new:.4f}')




# # best_auc = 0
# # best_model = None

# # hyperparams = [
# #     {"hidden_dim": 80, "learning_rate": 0.01, "num_layers": 6, "num_heads": 2},
# #     {"hidden_dim": 128, "learning_rate": 0.003, "num_layers": 7, "num_heads": 4},
# #     {"hidden_dim": 256, "learning_rate": 0.001, "num_layers": 8, "num_heads": 8},
# # ]

# # # for params in hyperparams:
# # #     print(f"Training with params: {params}")

# # #     model = GraphTransformer(in_dim, params["hidden_dim"], params["num_heads"], params["num_layers"])
# # #     predictor = LinkPredictor(params["hidden_dim"])
# # #     optimizer = torch.optim.Adam(itertools.chain(model.parameters(), predictor.parameters()), lr=params["learning_rate"])
    
# # #     for epoch in range(num_epochs):
# # #         loss = train_link_prediction(model, predictor, g, train_positive_edges, train_negative_edges, optimizer, loss_fn)
        
# # #         if epoch % 10 == 0:
# # #             val_auc, val_ap, _, _ = evaluate(model, predictor, g, valid_positive_edges, valid_negative_edges)
# # #             print(f"Epoch {epoch}: Loss={loss:.4f}, Val_AUC={val_auc:.4f}, Val_AP={val_ap:.4f}")

# # #             if val_auc > best_auc:
# # #                 best_auc = val_auc
# # #                 best_model = model.state_dict()

# # # # Save the best model
# # # torch.save(best_model, "best_graph_transformer.pth")
# # # print(f"Best model saved with AUC: {best_auc:.4f}")

# # # best_model = GraphTransformer(in_dim, hidden_dim, num_heads, num_layers)
# # best_model.load_state_dict(torch.load("best_graph_transformer.pth"))

# # test_auc, test_ap, _, _ = evaluate(best_model, predictor, g, test_positive_edges, test_negative_edges)
# # print(f"Best Model Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")



################################################################################
################################################################################


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

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
dgl.seed(seed)
dgl.random.seed(seed)

# Load data
(
    g, 
    train_positive_edges, train_negative_edges, 
    valid_positive_edges, valid_negative_edges, 
    test_positive_edges, test_negative_edges
) = load_data()

# Model, optimizer, and loss setup
in_dim = g.ndata['feat'].size(1)
hidden_dim = 80  
num_epochs = 300  
learning_rate = 0.00005  
num_layers = 6  
n_heads = 4  

# Initialize model and optimizer
model = GraphTransformer(in_dim, hidden_dim, n_heads, num_layers)
predictor = LinkPredictor(hidden_dim)
optimizer = torch.optim.Adam(
    itertools.chain(model.parameters(), predictor.parameters()), lr=learning_rate
)
loss_fn = nn.BCEWithLogitsLoss()

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

# Load best model for testing
best_model.load_state_dict(torch.load("best_model.pt"))

# Final evaluation on test links
test_auc, test_ap, lab, log = evaluate(best_model, predictor, g, test_positive_edges, test_negative_edges)

# Precision-recall and ROC curves
prec, rec, _ = precision_recall_curve(lab, log)
aupr_test_new = auc(rec, prec)
fpr, tpr, _ = roc_curve(lab, log)
auc_test_new = auc(fpr, tpr)

print(f'Final Test ROC AUC: {auc_test_new:.4f}')
print(f'Final Test AUPR: {aupr_test_new:.4f}')











