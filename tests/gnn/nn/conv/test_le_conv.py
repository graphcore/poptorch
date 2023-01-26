# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from torch_geometric.nn import LEConv
from conv_utils import conv_harness

out_channels = 16


def test_le_conv(dataset):
    in_channels = dataset.num_node_features
    conv = LEConv(in_channels, out_channels)

    conv_harness(conv, dataset)
