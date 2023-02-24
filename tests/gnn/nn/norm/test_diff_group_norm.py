# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import torch

from norm_utils import norm_harness

from torch_geometric.nn import DiffGroupNorm


def test_diff_group_norm():
    x = torch.randn(6, 16)

    norm = DiffGroupNorm(16, groups=4, lamda=0.01)
    assert str(norm) == 'DiffGroupNorm(16, groups=4)'

    out = norm_harness(norm, [x])
    assert out.size() == x.size()
