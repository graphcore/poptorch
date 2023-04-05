# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch
from torch_geometric.nn import GENConv

from conv_utils import conv_harness


@pytest.mark.parametrize('aggr', ['softmax', 'powermean'])
def test_gen_conv(aggr, dataset):
    in_channels = dataset.num_node_features

    conv = GENConv(in_channels,
                   32,
                   aggr,
                   edge_dim=16,
                   add_self_loops=False,
                   norm='layer')
    conv_harness(conv, dataset)

    x2 = torch.randn(dataset.x.shape)
    batch = ((dataset.x, x2), dataset.edge_index)
    conv_harness(conv, dataset, batch=batch)

    conv = GENConv((in_channels, in_channels),
                   32,
                   aggr,
                   add_self_loops=False,
                   norm='layer')
    conv_harness(conv, dataset, batch=batch)
