# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from torch_geometric.nn.aggr import Set2Set

from aggr_utils import aggr_harness


def test_set2set(dataloader):
    first_sample = next(iter(dataloader))
    channels = first_sample.num_node_features

    aggr = Set2Set(in_channels=channels, processing_steps=1)

    aggr_harness(aggr, first_sample.num_nodes, dataloader)
