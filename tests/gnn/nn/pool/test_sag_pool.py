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


@pytest.mark.parametrize('GNN', [GraphConv, GCNConv, GATConv, SAGEConv])
def test_sag_pooling(request, GNN):
    if GNN in (GraphConv, SAGEConv):
        pytest.skip(
            f"{request.node.nodeid}: Error: 'NotImplementedError: "
            "Could not run 'aten::_local_scalar_dense' with arguments from the"
            " 'Meta' backend'. Will be enabled after AFS-144 is fixed.")
    if GNN in (GCNConv, GATConv):
        pytest.skip(
            f"{request.node.nodeid}: Error: "
            "'torch_geometric/utils/loop.py:304 poptorch.poptorch_core.Error: "
            "In poptorch/source/dispatch_tracer/RegisterMetaOps.cpp.inc:163: "
            "'poptorch_cpp_error': Operations using aten::nonzero are "
            "unsupported because the output shape is determined by the tensor "
            "values. The IPU cannot support dynamic output shapes'. Will be "
            "enabled after AFS-145 is fixed.")

    in_channels = 16
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    pool1 = SAGPooling(in_channels, ratio=0.5, GNN=GNN)
    assert str(pool1) == (f'SAGPooling({GNN.__name__}, 16, '
                          f'ratio=0.5, multiplier=1.0)')
    out1 = pool_harness(pool1, [x, edge_index])
    assert out1[0].size() == (num_nodes // 2, in_channels)
    assert out1[1].size() == (2, 2)

    pool2 = SAGPooling(in_channels, ratio=None, GNN=GNN, min_score=0.1)
    assert str(pool2) == (f'SAGPooling({GNN.__name__}, 16, '
                          f'min_score=0.1, multiplier=1.0)')
    out2 = pool_harness(pool2, [x, edge_index])
    assert out2[0].size(0) <= x.size(0) and out2[0].size(1) == (16)
    assert out2[1].size(0) == 2 and out2[1].size(1) <= edge_index.size(1)

    pool3 = SAGPooling(in_channels, ratio=2, GNN=GNN)
    assert str(pool3) == (f'SAGPooling({GNN.__name__}, 16, '
                          f'ratio=2, multiplier=1.0)')
    out3 = pool_harness(pool3, [x, edge_index])
    assert out3[0].size() == (2, in_channels)
    assert out3[1].size() == (2, 2)
