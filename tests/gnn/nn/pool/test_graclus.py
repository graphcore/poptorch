# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from torch_geometric.nn import graclus
from torch_geometric.testing import withPackage

from pool_utils import pool_harness


@pytest.mark.skip(reason="TODO(AFS-245)")
@withPackage('torch_cluster')
def test_graclus():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    weight = torch.tensor([1., 1.])
    out = pool_harness(graclus, [edge_index, weight, 2])
    assert out.tolist() == [0, 0]
