# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn import (FastRGCNConv, GATConv, GCNConv, GINConv,
                                PNAConv, SAGEConv)


class GCN(torch.nn.Module):
    def __init__(self,
                 in_channels=0,
                 out_channels=0,
                 loss_fn=None,
                 disable_dropout=False):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 32, add_self_loops=False)
        self.conv2 = GCNConv(32, out_channels, add_self_loops=False)
        self.loss_fn = loss_fn
        self.disable_dropout = disable_dropout

    def forward(self, *args):
        x, edge_index = args
        x = F.relu(self.conv1(x, edge_index))
        if not self.disable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)

        if self.training:
            target = torch.ones_like(x)
            loss = self.loss_fn(x, target)
            return x, loss

        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels=0, loss_fn=None, disable_dropout=False):
        super().__init__()
        nn1 = Seq(Lin(in_channels, 32), ReLU(), Lin(32, 32))
        self.conv1 = GINConv(nn1, train_eps=True)
        nn2 = Seq(Lin(32, 32), ReLU(), Lin(32, 32))
        self.conv2 = GINConv(nn2, train_eps=True)
        self.loss_fn = loss_fn
        self.disable_dropout = disable_dropout

    def forward(self, *args):
        x, edge_index = args
        x = F.relu(self.conv1(x, edge_index))
        if not self.disable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)

        if self.training:
            target = torch.ones_like(x)
            loss = self.loss_fn(x, target)
            return x, loss

        return x


class GAT(torch.nn.Module):
    def __init__(self,
                 in_channels=0,
                 out_channels=0,
                 loss_fn=None,
                 disable_dropout=False):
        super().__init__()
        dropout_val = 0 if disable_dropout else 0.6
        self.conv1 = GATConv(in_channels,
                             8,
                             heads=8,
                             dropout=dropout_val,
                             add_self_loops=False)
        self.conv2 = GATConv(8 * 8,
                             out_channels,
                             dropout=dropout_val,
                             add_self_loops=False)
        self.loss_fn = loss_fn
        self.disable_dropout = disable_dropout

    def forward(self, *args):
        x, edge_index = args
        if not self.disable_dropout:
            x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        if not self.disable_dropout:
            x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)

        if self.training:
            target = torch.ones_like(x)
            loss = self.loss_fn(x, target)
            return x, loss

        return x


class RGCN(torch.nn.Module):
    def __init__(self,
                 in_channels=0,
                 out_channels=0,
                 num_relations=0,
                 loss_fn=None):

        super().__init__()
        self.conv1 = FastRGCNConv(in_channels,
                                  8,
                                  num_relations,
                                  num_bases=15,
                                  add_self_loops=False)
        self.conv2 = FastRGCNConv(8,
                                  out_channels,
                                  num_relations,
                                  num_bases=15,
                                  add_self_loops=False)
        self.loss_fn = loss_fn

    def forward(self, *args):
        edge_index, edge_type = args
        x = F.relu(self.conv1(None, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        x = F.log_softmax(x, dim=1)

        if self.training:
            target = torch.ones_like(x)
            loss = self.loss_fn(x, target)
            return x, loss

        return x


class PNA(torch.nn.Module):
    def __init__(self,
                 in_channels=0,
                 out_channels=0,
                 loss_fn=None,
                 disable_dropout=False,
                 degree=None):

        super().__init__()
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        self.conv = PNAConv(in_channels,
                            out_channels,
                            aggregators,
                            scalers,
                            deg=degree,
                            add_self_loops=False)
        self.loss_fn = loss_fn
        self.disable_dropout = disable_dropout

    def forward(self, *args):
        x, edge_index = args
        x = self.conv(x, edge_index)
        if not self.disable_dropout:
            x = F.dropout(x, training=self.training)

        if self.training:
            target = torch.ones_like(x)
            loss = self.loss_fn(x, target)
            return x, loss

        return x


class SAGE(torch.nn.Module):
    def __init__(self,
                 in_channels=0,
                 out_channels=0,
                 loss_fn=None,
                 disable_dropout=False):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels, add_self_loops=False)
        self.loss_fn = loss_fn
        self.disable_dropout = disable_dropout

    def forward(self, *args):
        x, edge_index = args
        x = self.conv(x, edge_index)
        if not self.disable_dropout:
            x = F.dropout(x, training=self.training)

        if self.training:
            target = torch.ones_like(x)
            loss = self.loss_fn(x, target)
            return x, loss

        return x
