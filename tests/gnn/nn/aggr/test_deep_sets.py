# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from torch_geometric.nn import DeepSetsAggregation, Linear

from aggr_utils import aggr_harness


def test_deep_sets_aggregation(dataloader):
    first_sample = next(iter(dataloader))
    channels = first_sample.num_node_features

    aggr = DeepSetsAggregation(
        local_nn=Linear(channels, channels * 2),
        global_nn=Linear(channels * 2, channels * 4),
    )
    aggr.reset_parameters()

    aggr_harness(aggr, first_sample.num_nodes, dataloader)
