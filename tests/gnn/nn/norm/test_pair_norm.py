# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch

from norm_utils import norm_harness

from torch_geometric.nn import PairNorm


@pytest.mark.parametrize('scale_individually', [False, True])
def test_pair_norm_no_batch(scale_individually):
    x = torch.randn(100, 16)

    norm = PairNorm(scale_individually=scale_individually)
    assert str(norm) == 'PairNorm()'

    out1 = norm_harness(norm, [x])
    assert out1.size() == (100, 16)


@pytest.mark.parametrize('scale_individually', [False, True])
def test_pair_norm(scale_individually):
    x = torch.randn(100, 16)
    batch = torch.zeros(100, dtype=torch.long)

    norm = PairNorm(scale_individually=scale_individually)
    assert str(norm) == 'PairNorm()'

    out1 = norm_harness(norm, [x])

    batch_size = 2
    out2 = norm_harness(norm, [
        torch.cat([x, x], dim=0),
        torch.cat([batch, batch + 1], dim=0), batch_size
    ])
    assert torch.allclose(out1, out2[:100], atol=1e-04)
    assert torch.allclose(out1, out2[100:], atol=1e-04)
