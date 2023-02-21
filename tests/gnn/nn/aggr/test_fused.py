# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
from torch_geometric.nn.aggr.fused import FusedAggregation

from aggr_utils import aggr_harness


@pytest.mark.parametrize('aggrs', [
    ['sum', 'mean', 'min', 'max', 'mul', 'var', 'std'],
    ['sum', 'min', 'max', 'mul', 'var', 'std'],
    ['min', 'max', 'mul', 'var', 'std'],
    ['mean', 'min', 'max', 'mul', 'var', 'std'],
    ['sum', 'min', 'max', 'mul', 'std'],
    ['mean', 'min', 'max', 'mul', 'std'],
    ['min', 'max', 'mul', 'std'],
])
def test_fused_aggregation(dataloader, aggrs):
    first_sample = next(iter(dataloader))
    in_channels = first_sample.num_node_features
    out_channels = in_channels * 2

    aggr = FusedAggregation(aggrs)
    post_proc = torch.nn.Linear(in_channels, out_channels)

    aggr_harness(aggr, first_sample.num_nodes, dataloader, post_proc)
