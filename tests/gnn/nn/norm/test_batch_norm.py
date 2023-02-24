# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch

from torch_geometric.nn import BatchNorm

from norm_utils import norm_harness


@pytest.mark.parametrize('conf', [True, False])
def test_batch_norm(conf):
    x = torch.randn(100, 16)

    norm = BatchNorm(16, affine=conf, track_running_stats=conf)
    assert str(norm) == 'BatchNorm(16)'

    out = norm_harness(norm, [x])
    assert out.size() == (100, 16)


def test_batch_norm_single_element(request):

    pytest.skip(f"{request.node.nodeid}: Error: 'BatchNorm Op \"101 "
                "(ai.graphcore.BatchNormalization:1)\" has 1 output(s) "
                "which means it is in inference mode, but its gradient "
                "Op is being created.")

    x = torch.randn(1, 16)

    norm = BatchNorm(16, track_running_stats=True, allow_single_element=True)
    assert str(norm) == 'BatchNorm(16)'

    out = norm_harness(norm, [x])
    assert torch.allclose(out, x)
