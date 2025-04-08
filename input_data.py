import pandas as pd 
import torch
import numpy as np
import dgl

path = '/home/netra-mobile/Desktop/GT-GRN_NEW/'

def load_data():
    raw_exp_file = path + 'Data/hESC/TFs+500/BL--ExpressionData.csv'
    train_file = path + 'Data/Train_validation_test/hESC 500/Train_set.csv'  # .../Demo/
    test_file = path +  'Data/Train_validation_test/hESC 500/Test_set.csv'
    val_file = path + 'Data/Train_validation_test/hESC 500/Validation_set.csv'
    exp_emb_file = path + 'Data/hESC/TFs+500/hESC_TF_500_emb.csv'
    global_file = path + 'Data/hESC/TFs+500/hESC_TF_global_500.csv'

    data_input = pd.read_csv(raw_exp_file, index_col=0)
    geneName = data_input.index

    train_data = pd.read_csv(train_file, index_col=0).values
    validation_data = pd.read_csv(val_file, index_col=0).values
    test_data = pd.read_csv(test_file, index_col=0).values
    exp_emb = pd.read_csv(exp_emb_file, header=None).to_numpy()
    global_features = pd.read_csv(global_file, header=None, index_col=0)

    global_features = global_features.loc[list(geneName)]


    train_positive_edges = torch.tensor(train_data[train_data[:, 2] == 1][:, :2].T)  # First two columns for positive labels
    train_negative_edges = torch.tensor(train_data[train_data[:, 2] == 0][:, :2].T)  # First two columns for negative labels

    self_loops = torch.arange(len(geneName))  
    self_loops = torch.stack([self_loops, self_loops], dim=0)  

    train_positive_edges = torch.cat([train_positive_edges, self_loops], dim=1)

    valid_positive_edges = torch.tensor(validation_data[validation_data[:, 2] == 1][:, :2].T)  # First two columns for positive labels
    valid_negative_edges = torch.tensor(validation_data[validation_data[:, 2] == 0][:, :2].T)  # First two columns for negative labels

    test_positive_edges = torch.tensor(test_data[test_data[:, 2] == 1][:, :2].T)  # First two columns for positive labels
    test_negative_edges = torch.tensor(test_data[test_data[:, 2] == 0][:, :2].T)  # First two columns for negative labels


    g = dgl.graph((train_positive_edges[0], train_positive_edges[1]), num_nodes=exp_emb.shape[0])
    g.ndata['feat'] = torch.tensor(exp_emb, dtype=torch.float32)
    g.ndata['global'] = torch.tensor(global_features.to_numpy(),  dtype=torch.float32)

    pos_enc_dim=g.ndata['feat'].shape[-1]
    g.ndata["PE"]=dgl.lap_pe(g,k=pos_enc_dim,padding=True)

    return g, train_positive_edges, train_negative_edges, valid_positive_edges, valid_negative_edges, test_positive_edges, test_negative_edges