# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
from torch_geometric.nn.aggr import SetTransformerAggregation

from aggr_utils import aggr_harness


@pytest.mark.skip(reason="TODO(AFS-279)")
def test_set_transformer_aggregation(dataloader):
    first_sample = next(iter(dataloader))
    channels = first_sample.num_node_features

    aggr = SetTransformerAggregation(channels, num_seed_points=2, heads=2)
    aggr.reset_parameters()

    aggr_harness(aggr, first_sample.num_nodes, dataloader, sorted_index=True)
