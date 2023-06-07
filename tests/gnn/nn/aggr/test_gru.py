# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from torch_geometric.nn import GRUAggregation

from aggr_utils import aggr_harness


def test_gru_aggregation(dataloader):
    first_sample = next(iter(dataloader))
    channels = first_sample.num_node_features

    aggr = GRUAggregation(channels, channels * 2)
    aggr.reset_parameters()

    aggr_harness(aggr, first_sample.num_nodes, dataloader, sorted_index=True)
