# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from torch_geometric.nn import MLPAggregation

from aggr_utils import aggr_harness


def test_mlp_aggregation(dataloader):
    first_sample = next(iter(dataloader))
    channels = first_sample.num_node_features

    aggr = MLPAggregation(
        in_channels=channels,
        out_channels=channels * 2,
        max_num_elements=first_sample.num_nodes,
        num_layers=1,
    )

    aggr_harness(aggr, first_sample.num_nodes, dataloader, sorted_index=True)
