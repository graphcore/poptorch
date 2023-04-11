# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GraphConv,
    SAGEConv,
    SAGPooling,
)

from pool_utils import pool_harness


@pytest.mark.skip(reason="The class is using filter_adj which produces "
                  "tensors with dynamic shapes. It is not supported "
                  "on Mk2.")
@pytest.mark.parametrize('GNN', [GraphConv, GCNConv, GATConv, SAGEConv])
def test_sag_pooling(GNN):
    conv_kwargs = {'add_self_loops': False}

    in_channels = 16
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    pool1 = SAGPooling(in_channels, ratio=0.5, GNN=GNN, **conv_kwargs)
    out1 = pool_harness(pool1, [x, edge_index])
    assert out1[0].size() == (num_nodes // 2, in_channels)
    assert out1[1].size() == (2, 2)

    pool2 = SAGPooling(in_channels,
                       ratio=None,
                       GNN=GNN,
                       min_score=0.1,
                       **conv_kwargs)
    out2 = pool_harness(pool2, [x, edge_index])
    assert out2[0].size(0) <= x.size(0) and out2[0].size(1) == (16)
    assert out2[1].size(0) == 2 and out2[1].size(1) <= edge_index.size(1)

    pool3 = SAGPooling(in_channels, ratio=2, GNN=GNN, **conv_kwargs)
    out3 = pool_harness(pool3, [x, edge_index])
    assert out3[0].size() == (2, in_channels)
    assert out3[1].size() == (2, 2)
