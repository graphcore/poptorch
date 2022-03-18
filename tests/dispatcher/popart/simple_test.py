#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import torch
import torch.nn as nn
from poptorch.experimental import IPUScope
import poptorch
import helpers


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_simple_test():
    input = torch.ones([10])

    with IPUScope([input]) as ipu:
        x = input + 5
        x = x * 3
        ipu.outputs(x)

    # pylint: disable=no-member
    helpers.assert_allequal(actual=ipu(input),
                            expected=torch.empty(10).fill_(18.0))


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_simple_conv():
    input = torch.ones([1, 5, 25, 25])

    conv = nn.Conv2d(5, 10, 5)

    with IPUScope([input], conv.named_parameters()) as ipu:
        x = conv(input)
        ipu.outputs(x)

    cpu = conv(input)
    ipu = ipu(input)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=cpu,
                            actual=ipu,
                            atol=1e-05,
                            rtol=1e-05,
                            equal_nan=True)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_tensor_constant():
    def f(x):
        return x + torch.tensor([1.0, 2.0, 3.0])

    input = torch.rand(3)
    with IPUScope([input]) as ipu:
        y = f(input)
        ipu.outputs(y)

    cpu = f(input)
    ipu = ipu(input)

    # pylint: disable=no-member
    helpers.assert_allequal(expected=cpu, actual=ipu)
