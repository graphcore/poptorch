# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from torch_geometric.nn import TransformerConv
from conv_utils import conv_harness

out_channels = 16


def test_transformer_conv(dataset):
    in_channels = dataset.num_node_features
    conv = TransformerConv(in_channels, out_channels, heads=2, beta=True)

    conv_harness(conv, dataset)
