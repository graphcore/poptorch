# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch

from norm_utils import norm_harness

from torch_geometric.nn import LayerNorm


@pytest.mark.parametrize('affine', [True, False])
@pytest.mark.parametrize('mode', ['graph', 'node'])
def test_layer_norm(affine, mode):

    if mode == 'graph':
        pytest.skip("TODO(AFS-242)")

    x = torch.randn(100, 16)

    norm = LayerNorm(16, affine=affine, mode=mode)
    assert str(norm) == f'LayerNorm(16, affine={affine}, mode={mode})'

    out1 = norm_harness(norm, [x])
    assert out1.size() == (100, 16)

    batch = torch.zeros(100, dtype=torch.int64)
    out2 = norm_harness(norm, [x, batch])
    assert out2.size() == (100, 16)

    out2 = norm_harness(
        norm, [torch.cat([x, x], dim=0),
               torch.cat([batch, batch + 1], dim=0)])
    assert out2.size() == (200, 16)
