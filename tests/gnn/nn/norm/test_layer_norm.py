# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch

from norm_utils import norm_harness

from torch_geometric.nn import LayerNorm


@pytest.mark.parametrize('affine', [True, False])
@pytest.mark.parametrize('mode', ['graph', 'node'])
def test_layer_norm(affine, mode):
    x = torch.randn(100, 16)

    norm = LayerNorm(16, affine=affine, mode=mode)

    norm_harness(norm, [x])

    batch = torch.zeros(100, dtype=torch.int64)
    batch_size = 1
    norm_harness(norm, [x, batch, batch_size])

    batch_size = 2
    norm_harness(norm, [
        torch.cat([x, x], dim=0),
        torch.cat([batch, batch + 1], dim=0), batch_size
    ])
