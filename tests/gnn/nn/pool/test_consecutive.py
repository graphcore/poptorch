# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch

from torch_geometric.nn.pool.consecutive import consecutive_cluster

from pool_utils import pool_harness


def test_consecutive_cluster(request):
    pytest.skip(
        f"{request.node.nodeid}: Error: 'Type inference failed for "
        "aten::_unique2 because the operator doesn't have an implementation "
        "for the Meta backend'. Will be enabled after AFS-136 is fixed.")

    src = torch.tensor([8, 2, 10, 15, 100, 1, 100])

    out, perm = pool_harness(consecutive_cluster, [src])
    assert out.tolist() == [2, 1, 3, 4, 5, 0, 5]
    assert perm.tolist() == [5, 1, 0, 2, 3, 6]
