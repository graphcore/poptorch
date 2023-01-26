# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn import NNConv
from conv_utils import conv_harness

out_channels = 16


def test_nn_conv(dataset):
    in_channels = dataset.num_node_features
    nn = Seq(Lin(3, 32), ReLU(), Lin(32, 8 * 32))
    conv = NNConv(in_channels, out_channels, nn=nn)

    value = torch.rand(dataset.num_edges, 3)
    batch = (dataset.x, dataset.edge_index, value)

    conv_harness(conv, dataset, batch=batch)
