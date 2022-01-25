# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import poptorch
import helpers

torch.manual_seed(42)
t = torch.randn(10)
min_tensor = torch.linspace(-0.5, 0.0, 10)
max_tensor = torch.linspace(0.0, 0.5, 10)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7.3 is not currently supported in MLIR.")
@pytest.mark.parametrize("t,min_params,max_params", [(t, None, 0.5),
                                                     (t, -0.5, None),
                                                     (t, -0.5, 0.5)])
def test_clamp(t, min_params, max_params):
    with poptorch.IPUScope([t],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        out = torch.clamp(t, min=min_params, max=max_params)
        ipu.outputs([out])

    ipu_result = ipu(t)
    cpu_result = torch.clamp(t, min=min_params, max=max_params)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result, actual=cpu_result)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7.3 is not currently supported in MLIR.")
@pytest.mark.parametrize("t,min_tensor,max_tensor",
                         [(t, min_tensor, max_tensor)])
def test_clampTensor(t, min_tensor, max_tensor):

    with poptorch.IPUScope([t, min_tensor, max_tensor],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        out = torch.clamp(t, min=min_tensor, max=max_tensor)
        ipu.outputs([out])

    ipu_result = ipu(t, min_tensor, max_tensor)
    cpu_result = torch.clamp(t, min=min_tensor, max=max_tensor)
    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result, actual=cpu_result)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7.3 is not currently supported in MLIR.")
@pytest.mark.parametrize("t,min_tensor", [(t, min_tensor)])
def test_clampTensor_min(t, min_tensor):
    with poptorch.IPUScope([t, min_tensor],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        out = torch.clamp(t, min=min_tensor)
        ipu.outputs([out])

    ipu_result = ipu(t, min_tensor)
    cpu_result = torch.clamp(t, min=min_tensor)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result, actual=cpu_result)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7.3 is not currently supported in MLIR.")
@pytest.mark.parametrize("t,max_tensor", [(t, max_tensor)])
def test_clampTensor_max(t, max_tensor):

    with poptorch.IPUScope([t, max_tensor],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        out = torch.clamp(t, max=max_tensor)
        ipu.outputs([out])

    ipu_result = ipu(t, max_tensor)
    cpu_result = torch.clamp(t, max=max_tensor)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result, actual=cpu_result)
