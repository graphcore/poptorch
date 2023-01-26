# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from torch_geometric.nn import ResGatedGraphConv
from conv_utils import conv_harness

out_channels = 16


def test_res_gated_graph_conv(dataset):
    in_channels = dataset.num_node_features

    conv = ResGatedGraphConv(in_channels, out_channels)
    conv_harness(conv, dataset)
