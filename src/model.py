import torch
import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# set dgl backend to pytorch
import os
os.environ['DGLBACKEND'] = 'pytorch'



# sparse attention module
class SparseMHA(nn.Module):
    def __init__(self,hidden_dim=80,num_heads=8):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads

        self.linear_q=nn.Linear(hidden_dim,hidden_dim)
        self.linear_k=nn.Linear(hidden_dim,hidden_dim)
        self.linear_v=nn.Linear(hidden_dim,hidden_dim)

        # projection of the output
        self.out_proj=nn.Linear(hidden_dim,hidden_dim)

    def forward(self,A,h):
        # A: [N,N], h: [N,hidden_dim]
        N=len(h)
        nh=self.num_heads
        dh=self.hidden_dim//nh 


        q=self.linear_q(h).reshape(N,dh,nh)
        k=self.linear_k(h).reshape(N,dh,nh)
        v=self.linear_k(h).reshape(N,dh,nh)

        attention_scores=dglsp.bsddmm(A,q,k.transpose(1,0)) # sparse [N,N,nh]

        attention_scores=attention_scores.softmax()         # (sparse) [N,N,nh]

        out=dglsp.bspmm(attention_scores,v) # [N,dh,nh]

        out=out.reshape(N,-1) # [N,hidden_dim]

        return self.out_proj(out)


class GTLayer(nn.Module):
    def __init__(self,hidden_dim=80,num_heads=8):
        super().__init__()

        self.attention=SparseMHA(hidden_dim,num_heads)
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads

        self.bn1=nn.BatchNorm1d(hidden_dim)
        self.bn2=nn.BatchNorm1d(hidden_dim)

        self.ffn=nn.Sequential(nn.Linear(hidden_dim,2*hidden_dim),
                              nn.ReLU(),
                              nn.Linear(2*hidden_dim,hidden_dim))
    def forward(self,A,h):

        h1=self.attention(A,h) # [N,hidden_dim]
        h=self.bn1(h+h1)

        h2=self.ffn(h)
        h=self.bn2(h+h2)

        return h
    

class GraphTransformer(nn.Module):
    def __init__(self, in_dim ,hidden_dim, num_heads, num_layers):
        super().__init__() 
        
        self.encoder = nn.Linear(in_dim, hidden_dim)
    
        self.pos_linear=nn.Linear(in_dim,hidden_dim)

        self.global_em = nn.Linear(in_dim, hidden_dim)

        # stack graph transformer layers
        self.layers=nn.ModuleList(
            [GTLayer(hidden_dim,num_heads) for _ in range(num_layers)]
        )

    def forward(self, g, X, pos_enc, global_emb):

        indices=torch.stack(g.edges())
        N=g.num_nodes()

        A=dglsp.spmatrix(indices,shape=(N,N))
        h=self.encoder(X) + self.pos_linear(pos_enc) + self.global_em(global_emb)
        

        for layer in self.layers:
            h=layer(A,h)

        return h
