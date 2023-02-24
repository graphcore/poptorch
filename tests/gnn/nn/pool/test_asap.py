# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from torch_geometric.nn import ASAPooling, GCNConv, GraphConv

from pool_utils import pool_harness


def test_asap(request):
    pytest.skip(
        f"{request.node.nodeid}: Error: 'The IPU cannot support dynamic output"
        " shapes'. Will be enabled after AFS-145 is fixed.")

    in_channels = 16
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    for GNN in [GraphConv, GCNConv]:
        pool = ASAPooling(in_channels,
                          ratio=0.5,
                          GNN=GNN,
                          add_self_loops=False)
        assert pool.__repr__() == ('ASAPooling(16, ratio=0.5)')
        out = pool_harness(pool, [x, edge_index])
        assert out[0].size() == (num_nodes // 2, in_channels)
        assert out[1].size() == (2, 2)

        pool = ASAPooling(in_channels, ratio=0.5, GNN=GNN, add_self_loops=True)
        assert pool.__repr__() == ('ASAPooling(16, ratio=0.5)')
        out = pool_harness(pool, [x, edge_index])
        assert out[0].size() == (num_nodes // 2, in_channels)
        assert out[1].size() == (2, 4)

        pool = ASAPooling(in_channels, ratio=2, GNN=GNN, add_self_loops=False)
        assert pool.__repr__() == ('ASAPooling(16, ratio=2)')
        out = pool_harness(pool, [x, edge_index])
        assert out[0].size() == (2, in_channels)
        assert out[1].size() == (2, 2)
