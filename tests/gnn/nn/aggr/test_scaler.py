# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
from torch_geometric.nn import DegreeScalerAggregation

from aggr_utils import aggr_harness


@pytest.mark.parametrize('scaler',
                         [['identity'], ['amplification'], ['attenuation'],
                          ['linear'], ['inverse_linear']])
@pytest.mark.parametrize('train_norm', [True, False])
def test_degree_scaler_aggregation(dataloader, scaler, train_norm):

    first_sample = next(iter(dataloader))
    in_channels = first_sample.num_node_features
    out_channels = in_channels * 2

    deg = torch.tensor([2, 5, 3, 1, 2, 3, 4, 1, 2, 0])

    basic_aggrs = ['mean', 'sum', 'max']
    aggr = DegreeScalerAggregation(basic_aggrs,
                                   scaler,
                                   deg,
                                   train_norm=train_norm)
    post_proc = torch.nn.Linear(in_channels * len(basic_aggrs),
                                out_channels * len(basic_aggrs))

    aggr_harness(aggr, first_sample.num_nodes, dataloader, post_proc)
