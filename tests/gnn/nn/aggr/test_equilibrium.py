# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
from torch_geometric.nn import EquilibriumAggregation

from aggr_utils import aggr_harness


@pytest.mark.parametrize('grad_iter', [0, 1, 5])
def test_equilibrium(dataloader, grad_iter):
    pytest.skip("TODO(AFS-162)")

    first_sample = next(iter(dataloader))
    channels = first_sample.num_node_features

    aggr = EquilibriumAggregation(channels,
                                  channels // 2,
                                  num_layers=[10, 10],
                                  grad_iter=grad_iter)

    aggr_harness(aggr, first_sample.num_nodes, dataloader)
