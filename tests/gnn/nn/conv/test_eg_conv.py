# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
from torch_geometric.nn import EGConv
from conv_utils import conv_harness

conv_kwargs_list = [
    {
        "add_self_loops": False
    },
    {
        "add_self_loops": False,
        "aggregators": ["max", "min"]
    },
]


@pytest.mark.parametrize('conv_kwargs', conv_kwargs_list)
def test_eg_conv(dataset, conv_kwargs):
    in_channels = dataset.num_node_features
    conv = EGConv(in_channels, 32, **conv_kwargs)

    conv_harness(conv, dataset)
