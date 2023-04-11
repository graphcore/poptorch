# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
from torch_geometric.nn import LSTMAggregation

from aggr_utils import aggr_harness


@pytest.mark.skip(reason="TODO(AFS-279)")
def test_lstm_aggregation(dataloader):
    first_sample = next(iter(dataloader))
    channels = first_sample.num_node_features

    aggr = LSTMAggregation(channels, channels * 2)

    aggr_harness(aggr, first_sample.num_nodes, dataloader, sorted_index=True)
