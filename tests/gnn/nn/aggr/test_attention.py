# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch
from torch_geometric.nn import MLP
from torch_geometric.nn.aggr import AttentionalAggregation

from aggr_utils import aggr_harness


def test_attentional_aggregation(dataloader):
    first_sample = next(iter(dataloader))
    in_channels = first_sample.num_node_features
    out_channels = in_channels * 2

    gate_nn = MLP([in_channels, 1], act='relu')
    nn = MLP([in_channels, in_channels], act='relu')
    aggr = AttentionalAggregation(gate_nn, nn)
    post_proc = torch.nn.Linear(in_channels, out_channels)

    aggr_harness(aggr,
                 first_sample.num_nodes,
                 dataloader,
                 post_proc,
                 atol=1e-3,
                 rtol=5e-3)
