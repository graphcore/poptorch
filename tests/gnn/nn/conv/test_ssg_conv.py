# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import torch
from torch_geometric.nn import SSGConv

from conv_utils import conv_harness


def test_ssg_conv(dataset):
    in_channels = dataset.num_node_features
    out_channels = 32

    conv = SSGConv(in_channels,
                   out_channels,
                   alpha=0.1,
                   K=10,
                   add_self_loops=False)
    conv_harness(conv, dataset)

    value = torch.rand(dataset.num_edges)
    conv_harness(conv, batch=(dataset.x, dataset.edge_index, value))
