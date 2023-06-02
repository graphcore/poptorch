# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest

import torch
from torch_geometric.nn import SimpleConv

from conv_utils import conv_harness


@pytest.mark.parametrize('combine_root', ['sum', 'cat', 'self_loop', None])
def test_simple_conv(dataset, combine_root):
    in_channels = dataset.num_node_features
    out_channels = 64

    if combine_root == 'cat':
        in_channels = in_channels * 2

    lin = torch.nn.Linear(in_channels, out_channels)
    conv = SimpleConv(combine_root=combine_root)

    conv_harness(conv, dataset, post_proc=lin)
