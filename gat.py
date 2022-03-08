"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
            if l == self.num_layers-1:
                embed = h
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits, embed

# linear
class LNet(nn.Module):
    def __init__(self, emb_size, out_size):
        super(LNet, self).__init__()
        self.cls = nn.Linear(emb_size, out_size)

    def forward(self, x):
        return self.cls(x)


# MLP
class FC(nn.Module):
    def __init__(self, in_size, out_size):
        super(FC, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.linear(x))

class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super(MLP, self).__init__()
        self.fc = FC(in_size, mid_size)
        self.cls = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.cls(self.fc(x))
