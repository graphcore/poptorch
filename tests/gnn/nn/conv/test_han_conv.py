# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from torch_geometric.nn import HANConv

from conv_utils import hetero_conv_harness, random_heterodata


def test_han_conv():
    data, in_channels = random_heterodata()
    metadata = data.metadata()

    conv = HANConv(in_channels, 16, metadata, heads=2, add_self_loops=False)
    hetero_conv_harness(conv, data, 'author')


def test_han_conv_lazy():
    data, _ = random_heterodata()
    metadata = data.metadata()

    conv = HANConv(-1, 16, metadata, heads=2, add_self_loops=False)
    _ = conv(data.x_dict, data.edge_index_dict)
    hetero_conv_harness(conv, data, 'author')
