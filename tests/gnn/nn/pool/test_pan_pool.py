# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from torch_geometric.nn import PANConv, PANPooling

from pool_utils import op_harness


def test_pan_pooling(request):
    pytest.skip(
        f"{request.node.nodeid}: Error: "
        "'torch_geometric/nn/pool/pan_pool.py:93 RuntimeError: Expected all "
        "tensors to be on the same device, but found at least two devices, "
        "ipu:0 and cpu!'. Will be enabled after AFS-142 is fixed.")

    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, 16))

    conv = PANConv(16, 32, filter_size=2)
    pool = PANPooling(32, ratio=0.5)
    assert str(pool) == 'PANPooling(32, ratio=0.5, multiplier=1.0)'

    x, M = conv(x, edge_index)
    h, edge_index, edge_weight, _, perm, score = op_harness(pool, [x, M])

    assert h.size() == (2, 32)
    assert edge_index.size() == (2, 4)
    assert edge_weight.size() == (4, )
    assert perm.size() == (2, )
    assert score.size() == (2, )
