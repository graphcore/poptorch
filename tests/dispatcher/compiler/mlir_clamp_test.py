# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import poptorch
from poptorch.experimental import IPUContext
import helpers


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7.3 is not currently supported in MLIR.")
@pytest.mark.parametrize("min_param,max_param", [(None, 0.5), (-0.5, None),
                                                 (-0.5, 0.5)])
def test_clamp(min_param, max_param):
    torch.manual_seed(42)
    t = torch.randn(10)

    ipu_result = IPUContext(torch.clamp)(t, min_param, max_param)
    cpu_result = torch.clamp(t, min_param, max_param)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=cpu_result, actual=ipu_result)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7.3 is not currently supported in MLIR.")
@pytest.mark.parametrize("clamp_min,clamp_max", [(True, True), (True, False),
                                                 (False, True)])
def test_clampTensor(clamp_min, clamp_max):
    torch.manual_seed(42)
    t = torch.randn(10)
    min_tensor = torch.linspace(-0.5, 0.0, 10) if clamp_min else None
    max_tensor = torch.linspace(0.0, 0.5, 10) if clamp_max else None

    ipu_result = IPUContext(torch.clamp)(t, min=min_tensor, max=max_tensor)
    cpu_result = torch.clamp(t, min=min_tensor, max=max_tensor)
    # pylint: disable=no-member
    helpers.assert_allclose(expected=cpu_result, actual=ipu_result)
