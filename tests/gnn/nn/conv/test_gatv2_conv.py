# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
from torch_geometric.nn import GATv2Conv
from conv_utils import conv_harness

out_channels = 32
conv_kwargs_list = [
    {
        'edge_dim': None
    },
    {
        'edge_dim': 1,
        'fill_value': 0.5
    },
    {
        'edge_dim': 1,
        'fill_value': 'mean'
    },
    {
        'edge_dim': 4,
        'fill_value': 0.5
    },
    {
        'edge_dim': 4,
        'fill_value': 'mean'
    },
]


@pytest.mark.parametrize('conv_kwargs', conv_kwargs_list)
def test_gatv2_conv(dataset, conv_kwargs):
    in_channels = dataset.num_node_features
    conv_kwargs["add_self_loops"] = False
    conv = GATv2Conv(in_channels, out_channels, heads=2, **conv_kwargs)

    conv_harness(conv, dataset, atol=1e-4, rtol=1e-3)
