# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
from torch_geometric.nn import MultiAggregation

from aggr_utils import aggr_harness


@pytest.mark.parametrize('mode', [
    'cat', 'proj', 'attn', 'sum', 'mean', 'max', 'min', 'logsumexp', 'std',
    'var'
])
def test_multi_aggr(dataloader, mode):
    first_sample = next(iter(dataloader))
    in_channels = first_sample.num_node_features
    out_channels = in_channels * 2

    mode_kwargs = None
    if mode == 'proj':
        mode_kwargs = dict(in_channels=in_channels, out_channels=in_channels)
    elif mode == 'attn':
        mode_kwargs = dict(in_channels=in_channels,
                           out_channels=in_channels,
                           num_heads=in_channels // 4)

    aggrs = ['mean', 'sum', 'max']
    aggr = MultiAggregation(aggrs, mode=mode, mode_kwargs=mode_kwargs)
    aggr.reset_parameters()

    if mode == 'cat':
        # The 'cat' combine mode will expand the output dimensions
        # the number of aggregators.
        in_channels = in_channels * len(aggrs)
        out_channels = out_channels * len(aggrs)

    post_proc = torch.nn.Linear(in_channels, out_channels)

    aggr_harness(aggr,
                 first_sample.num_nodes,
                 dataloader,
                 post_proc,
                 atol=1e-4)
