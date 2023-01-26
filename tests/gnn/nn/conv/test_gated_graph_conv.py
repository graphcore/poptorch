# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from torch_geometric.nn import GatedGraphConv
from conv_utils import conv_harness

out_channels = 32


def test_gated_graph_conv(dataset):
    in_channels = dataset.num_node_features
    conv = GatedGraphConv(in_channels, num_layers=3)

    conv_harness(conv, dataset)
