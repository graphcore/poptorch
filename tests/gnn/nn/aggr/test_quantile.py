# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
from torch_geometric.nn import MedianAggregation, QuantileAggregation

from aggr_utils import aggr_harness


@pytest.mark.parametrize('q', [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
@pytest.mark.parametrize('interpolation', QuantileAggregation.interpolations)
@pytest.mark.skip(reason="TODO(AFS-275, AFS-277, AFS-278)")
def test_quantile_aggregation(dataloader, q, interpolation):
    first_sample = next(iter(dataloader))
    in_channels = first_sample.num_node_features
    out_channels = in_channels * 2

    aggr = QuantileAggregation(q=q, interpolation=interpolation)
    post_proc = torch.nn.Linear(in_channels, out_channels)

    aggr_harness(aggr,
                 first_sample.num_nodes,
                 dataloader,
                 post_proc,
                 sorted_index=True)


@pytest.mark.skip(reason="TODO(AFS-275, AFS-277, AFS-278)")
def test_median_aggregation(dataloader):
    first_sample = next(iter(dataloader))
    in_channels = first_sample.num_node_features
    out_channels = in_channels * 2

    aggr = MedianAggregation()
    post_proc = torch.nn.Linear(in_channels, out_channels)

    aggr_harness(aggr, first_sample.num_nodes, dataloader, post_proc)
