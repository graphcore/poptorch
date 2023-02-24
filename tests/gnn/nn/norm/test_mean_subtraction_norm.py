# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch

from norm_utils import norm_harness

from torch_geometric.nn import MeanSubtractionNorm


def test_mean_subtraction_norm_no_batch():
    x = torch.randn(6, 16)

    norm = MeanSubtractionNorm()
    assert str(norm) == 'MeanSubtractionNorm()'

    out = norm_harness(norm, [x])
    assert out.size() == (6, 16)
    assert torch.allclose(out.mean(), torch.tensor(0.), atol=1e-04)


def test_mean_subtraction_norm(request):
    pytest.skip(
        f"{request.node.nodeid}: Error: 'Could not run "
        "'aten::_local_scalar_dense' with arguments from the 'Meta' backend'."
        " Will be enabled after AFS-144 is fixed.")

    x = torch.randn(6, 16)
    batch = torch.tensor([0, 0, 1, 1, 1, 2])

    norm = MeanSubtractionNorm()
    assert str(norm) == 'MeanSubtractionNorm()'

    out = norm_harness(norm, [x, batch])
    assert out.size() == (6, 16)
    assert torch.allclose(out[0:2].mean(), torch.tensor(0.), atol=1e-04)
    assert torch.allclose(out[0:2].mean(), torch.tensor(0.), atol=1e-04)
