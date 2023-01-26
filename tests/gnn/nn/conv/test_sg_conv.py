# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from torch_geometric.nn import SGConv
from conv_utils import conv_harness

out_channels = 16


def test_sg_conv(dataset):
    in_channels = dataset.num_node_features
    conv = SGConv(in_channels, out_channels, K=10, add_self_loops=False)

    conv_harness(conv, dataset)


def test_sg_weights_conv(dataset):
    in_channels = dataset.num_node_features
    conv = SGConv(in_channels, out_channels, K=10, add_self_loops=False)

    batch = (dataset.x, dataset.edge_index, dataset.edge_weight)
    conv_harness(conv, dataset, batch=batch)
