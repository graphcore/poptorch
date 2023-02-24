# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch

from norm_utils import norm_harness

from torch_geometric.nn import InstanceNorm


@pytest.mark.parametrize('conf', [True, False])
def test_instance_norm(request, conf):

    pytest.skip(
        f"{request.node.nodeid}: Error: 'Could not run "
        "'aten::_local_scalar_dense' with arguments from the 'Meta' backend'."
        " Will be enabled after AFS-144 is fixed.")

    batch = torch.zeros(100, dtype=torch.long)

    x1 = torch.randn(100, 16)
    x2 = torch.randn(100, 16)

    norm1 = InstanceNorm(16, affine=conf, track_running_stats=conf)
    norm2 = InstanceNorm(16, affine=conf, track_running_stats=conf)
    assert str(norm1) == 'InstanceNorm(16)'

    out1 = norm_harness(norm1, [x1])
    out2 = norm_harness(norm2, [x1, batch])
    assert out1.size() == (100, 16)
    assert torch.allclose(out1, out2, atol=1e-7)
    if conf:
        assert torch.allclose(norm1.running_mean, norm2.running_mean)
        assert torch.allclose(norm1.running_var, norm2.running_var)

    out1 = norm_harness(norm1, [x2])
    out2 = norm_harness(norm2, [x2, batch])
    assert torch.allclose(out1, out2, atol=1e-7)
    if conf:
        assert torch.allclose(norm1.running_mean, norm2.running_mean)
        assert torch.allclose(norm1.running_var, norm2.running_var)

    norm1.eval()
    norm2.eval()

    out1 = norm_harness(norm1, [x1])
    out2 = norm_harness(norm2, [x1, batch])
    assert torch.allclose(out1, out2, atol=1e-7)

    out1 = norm_harness(norm1, [x2])
    out2 = norm_harness(norm2, [x2, batch])
    assert torch.allclose(out1, out2, atol=1e-7)

    out1 = norm_harness(norm2, [x1])
    out2 = norm_harness(norm2, [x2])
    out3 = norm_harness(
        norm2, [torch.cat([x1, x2], dim=0),
                torch.cat([batch, batch + 1])])
    assert torch.allclose(out1, out3[:100], atol=1e-7)
    assert torch.allclose(out2, out3[100:], atol=1e-7)
