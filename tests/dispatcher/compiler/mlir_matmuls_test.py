#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import helpers
import poptorch

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

    with poptorch.IPUScope([t1, t2],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        out = torch.matmul(t1, t2)
        ipu.outputs([out])

    ipu_result = ipu(t1, t2)

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
