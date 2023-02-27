# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from torch_geometric.nn import AntiSymmetricConv
from torch_geometric.nn.conv import GCNConv

from conv_utils import conv_harness


def test_antisymmetric_conv(dataset):
    in_channels = dataset.num_node_features
    phi = GCNConv(in_channels, in_channels, bias=False, add_self_loops=False)
    conv = AntiSymmetricConv(in_channels, phi=phi)

    conv_harness(conv, dataset)
