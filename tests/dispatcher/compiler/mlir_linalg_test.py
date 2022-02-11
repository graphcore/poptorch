#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import helpers
import poptorch
from poptorch.experimental import IPUContext

to_test = [
    # Vector-Vector
    [[12], [12]],

    # Matrix-Vector
    [[4, 5], [5]],

    # Matrix-matrix
    [[10, 5], [5, 7]],

    # Batch-Matrix-Batch-matrix
    [[6, 10, 5], [6, 5, 7]],

    # Batch-Matrix-Batch-matrix broadcast
    [[3, 1, 10, 5], [1, 6, 5, 7]]
]


@pytest.mark.parametrize("size", to_test)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_matmul(size):
    torch.manual_seed(42)

    t1 = torch.randn(size[0])
    t2 = torch.randn(size[1])

    ipu_result = IPUContext(torch.matmul)(t1, t2)
    cpu_result = torch.matmul(t1, t2)

    if cpu_result.numel() == 1:
        cpu_result = cpu_result.reshape((1))

    assert ipu_result.size() == cpu_result.size()

    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result,
                            actual=cpu_result,
                            atol=1e-05,
                            rtol=1e-05,
                            equal_nan=True)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize(
    "params",
    [
        # input_shape, beta, alpha
        ((3, 7), 1.0, 1.0),
        ((3, 1), 1.0, 0.75),
        ((1, 7), 0.75, 1.0),
        ((1), 0.75, 0.75),
    ])
def test_addmm(params):
    torch.manual_seed(42)

    input_shape, beta, alpha = params

    t1 = torch.randn(input_shape)
    t2 = torch.randn(3, 5)
    t3 = torch.randn(5, 7)

    def addmm(x1, x2, x3):
        return torch.addmm(x1, x2, x3, beta=beta, alpha=alpha)

    cpu_result = addmm(t1, t2, t3)
    ipu_result = IPUContext(addmm)(t1, t2, t3)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=cpu_result, actual=ipu_result)
