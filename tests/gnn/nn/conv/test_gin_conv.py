# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn import GINConv, GINEConv
from conv_utils import conv_harness

out_channels = 32


def test_gin_conv(dataset):
    in_channels = dataset.num_node_features
    nn = Seq(Lin(in_channels, 32), ReLU(), Lin(32, 32))
    conv = GINConv(nn, train_eps=True)

    conv_harness(conv, dataset)


def test_gine_conv(dataset):
    in_channels = dataset.num_node_features
    nn = Seq(Lin(in_channels, 32), ReLU(), Lin(32, 32))

    conv = GINEConv(nn, train_eps=True)

    value = torch.randn(dataset.num_edges, 16)
    batch = (dataset.x, dataset.edge_index, value)

    conv_harness(conv, dataset, batch=batch)
