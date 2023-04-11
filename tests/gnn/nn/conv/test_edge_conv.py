# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn import DynamicEdgeConv, EdgeConv
from conv_utils import conv_harness

out_channels = 32


def test_edge_conv(dataset):
    in_channels = dataset.num_node_features
    nn = Seq(Lin(in_channels * 2, in_channels), ReLU(),
             Lin(in_channels, out_channels))
    conv = EdgeConv(nn)

    conv_harness(conv, dataset)


def test_dynamic_edge_conv(dataset):
    in_channels = dataset.num_node_features
    nn = Seq(Lin(in_channels * 2, in_channels), ReLU(),
             Lin(in_channels, out_channels))
    conv = DynamicEdgeConv(nn, k=2)

    conv_harness(conv, dataset, batch=(dataset.x, ))
