# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
from torch_geometric.nn.aggr import GraphMultisetTransformer

from aggr_utils import aggr_harness


def test_graph_multiset_transformer(dataloader):
    pytest.skip("TODO(AFS-162)")

    first_sample = next(iter(dataloader))
    channels = first_sample.num_node_features

    aggr = GraphMultisetTransformer(channels, k=2, heads=2)
    aggr.reset_parameters()

    aggr_harness(aggr, first_sample.num_nodes, dataloader, sorted_index=True)
