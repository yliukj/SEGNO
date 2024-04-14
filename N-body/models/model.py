import torch
from torch import nn
from models.models.gcl import SEGNO_GCL


class SEGNO(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0,
                 recurrent=False, norm_diff=False, tanh=False, invariant=True, norm_vel=True):
        super(SEGNO, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.invariant = invariant
        self.norm_vel = norm_vel
        self.sigmoid = nn.Sigmoid()
        self.module = SEGNO_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                        act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent,
                                                        norm_diff=norm_diff, tanh=tanh, norm_vel=norm_vel)
        self.to(self.device)

    def forward(self, his, x, edges, v, edge_attr):
        h = self.embedding(his)
        for i in range(self.n_layers):
            h, x, v, _ = self.module(h, edges, x, v, v, edge_attr=edge_attr)
            
        return x, h
        