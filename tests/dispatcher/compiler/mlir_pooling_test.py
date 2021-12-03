#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
from torch import nn
import pytest
import helpers
import poptorch
from poptorch.enums import Compiler


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize(
    "params",
    [
        # op, spatial dims
        (nn.AdaptiveAvgPool1d, 1),
        (nn.AdaptiveAvgPool2d, 2),
        (nn.AdaptiveAvgPool3d, 3),
    ])
def test_adaptive_avg_pool(params):
    torch.manual_seed(42)
    # AdaptiveAvgPool1d: [1, 2, 4]       -> [1, 2, 2]
    # AdaptiveAvgPool2d: [1, 2, 4, 6]    -> [1, 2, 2, 3]
    # AdaptiveAvgPool3d: [1, 2, 4, 6, 8] -> [1, 2, 2, 3, 4]
    # TODO(T31335): Match PyTorch's implementation so that we can test cases where
    #               input dims are not divisible by corresponding output dims

    pool_op, n_output_dims = params

    shape = [1, 2]
    shape.extend([2 * i + 4 for i in range(n_output_dims)])

    t = torch.randn(shape)
    output_size = [i + 2 for i in range(n_output_dims)]

    pool = pool_op(output_size)
    # Run pytorch native on CPU.
    torch_out = pool(t)

    # Run on IPU.
    with poptorch.IPUScope([t], compile_using=Compiler.MLIR) as ipu:
        ipu.outputs([pool(t)])

    # pylint: disable=no-member
    helpers.assert_allclose(actual=ipu(t), expected=torch_out)
