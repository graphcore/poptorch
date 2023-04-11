# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
from torch_geometric.nn.aggr import SortAggregation

from aggr_utils import aggr_harness


@pytest.mark.skip(reason="TODO(AFS-279, AFS-277)")
def test_sort_aggregation(dataloader):
    first_sample = next(iter(dataloader))
    in_channels = first_sample.num_node_features
    out_channels = in_channels * 2

    k = 5
    aggr = SortAggregation(k=k)
    post_proc = torch.nn.Linear(k * in_channels, k * out_channels)

    aggr_harness(aggr,
                 first_sample.num_nodes,
                 dataloader,
                 post_proc,
                 sorted_index=True)
