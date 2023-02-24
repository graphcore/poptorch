# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch

from norm_utils import norm_harness

from torch_geometric.nn import GraphNorm


def test_graph_norm(request):

    pytest.skip(
        f"{request.node.nodeid}: Error: 'Could not run "
        "'aten::_local_scalar_dense' with arguments from the 'Meta' backend'."
        " Will be enabled after AFS-144 is fixed.")

    torch.manual_seed(42)
    x = torch.randn(200, 16)
    batch = torch.arange(4).view(-1, 1).repeat(1, 50).view(-1)

    norm = GraphNorm(16)
    assert str(norm) == 'GraphNorm(16)'

    out = norm_harness(norm, [x])
    assert out.size() == (200, 16)
    assert torch.allclose(out.mean(dim=0), torch.zeros(16), atol=1e-6)
    assert torch.allclose(out.std(dim=0, unbiased=False),
                          torch.ones(16),
                          atol=1e-6)

    out = norm_harness(norm, [x, batch])
    assert out.size() == (200, 16)
    assert torch.allclose(out[:50].mean(dim=0), torch.zeros(16), atol=1e-6)
    assert torch.allclose(out[:50].std(dim=0, unbiased=False),
                          torch.ones(16),
                          atol=1e-6)
