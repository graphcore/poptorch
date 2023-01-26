# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from torch_geometric.nn import ARMAConv
from conv_utils import conv_harness

out_channels = 32


def test_arma_conv(dataset):
    in_channels = dataset.num_node_features
    conv = ARMAConv(in_channels, out_channels, num_stacks=8, num_layers=4)

    conv_harness(conv, dataset)
