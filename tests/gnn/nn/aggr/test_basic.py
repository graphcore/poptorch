# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
from torch_geometric.nn import (
    MaxAggregation,
    MeanAggregation,
    MinAggregation,
    MulAggregation,
    PowerMeanAggregation,
    SoftmaxAggregation,
    StdAggregation,
    SumAggregation,
    VarAggregation,
)

from aggr_utils import aggr_harness


@pytest.mark.parametrize('Aggregation', [
    MeanAggregation,
    SumAggregation,
    MaxAggregation,
    MinAggregation,
    MulAggregation,
    VarAggregation,
    StdAggregation,
])
def test_basic_aggregation(dataloader, Aggregation):
    first_sample = next(iter(dataloader))
    in_channels = first_sample.num_node_features
    out_channels = in_channels * 2

    aggr = Aggregation()
    post_proc = torch.nn.Linear(in_channels, out_channels)

    aggr_harness(aggr, first_sample.num_nodes, dataloader, post_proc)


@pytest.mark.parametrize('Aggregation', [
    SoftmaxAggregation,
    PowerMeanAggregation,
])
@pytest.mark.parametrize('learn', [True, False])
def test_gen_aggregation(dataloader, Aggregation, learn):
    first_sample = next(iter(dataloader))
    in_channels = first_sample.num_node_features
    out_channels = in_channels * 2

    aggr = Aggregation(learn=learn)
    post_proc = torch.nn.Linear(in_channels, out_channels)

    if learn is True and isinstance(aggr, PowerMeanAggregation):
        pytest.skip("TODO(AFS-178)")

    aggr_harness(aggr, first_sample.num_nodes, dataloader, post_proc)


@pytest.mark.parametrize('Aggregation', [
    SoftmaxAggregation,
    PowerMeanAggregation,
])
def test_learnable_channels_aggregation(dataloader, Aggregation):
    first_sample = next(iter(dataloader))
    channels = first_sample.num_node_features

    aggr = Aggregation(learn=True, channels=channels)

    if isinstance(aggr, PowerMeanAggregation):
        pytest.skip("TODO(AFS-178)")

    aggr_harness(aggr, first_sample.num_nodes, dataloader)
