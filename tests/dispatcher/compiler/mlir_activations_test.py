#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import torch.nn.functional as F
import pytest
import helpers
import poptorch


# It's a bit of a pain to express these in the list without a python func inbetween.
def relu_inplace(x):
    x.relu_()
    return x


def tanh_inplace(x):
    x.tanh_()
    return x


def sigmoid_inplace(x):
    x.sigmoid_()
    return x


def silu_inplace(x):
    F.silu(x, inplace=True)
    return x


def hardsigmoid_inplace(x):
    F.hardsigmoid(x, inplace=True)
    return x


def hardswish_inplace(x):
    F.hardswish(x, inplace=True)
    return x


activation_functions = [
    F.relu, F.tanh, F.sigmoid, F.gelu, F.hardsigmoid, F.silu, F.hardswish,
    relu_inplace, tanh_inplace, sigmoid_inplace, silu_inplace,
    hardsigmoid_inplace, hardswish_inplace
]


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize("op", activation_functions)
def test_activations(op):
    torch.manual_seed(42)

    input = torch.randn([2, 10, 4, 2])

    with poptorch.IPUScope([input],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        out = op(input)
        ipu.outputs([out])

    ipu_result = ipu(input)
    cpu_result = op(input)

    assert ipu_result.size() == cpu_result.size()

    tol = [0.01, 1e-3] if op is F.gelu else [1e-4, 1e-7]

    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result,
                            actual=cpu_result,
                            atol=tol[0],
                            rtol=tol[1],
                            equal_nan=True)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_sigmoid(dim):
    torch.manual_seed(42)

    input1 = torch.randn([2, 2, 4, 5])

    with poptorch.IPUScope([input1],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        out = F.softmax(input1, dim)
        ipu.outputs([out])

    cpu_result = F.softmax(input1, dim)
    ipu_result = ipu(input1)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result,
                            actual=cpu_result,
                            equal_nan=True)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_logsoftmax_forward(dim):
    torch.manual_seed(42)
    input1 = torch.randn([2, 2, 4, 5])
    with poptorch.IPUScope([input1],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        out = F.log_softmax(input1, dim)
        ipu.outputs([out])

    cpu_result = F.log_softmax(input1, dim)
    ipu_result = ipu(input1)
    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result,
                            actual=cpu_result,
                            equal_nan=True)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_logsoftmax_backward(dim):
    torch.manual_seed(42)
    input1 = torch.nn.parameter.Parameter(torch.randn([2, 2, 4, 5]))
    with poptorch.IPUScope([input1.data],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        out = F.log_softmax(input1, dim)
        loss = torch.sum(out)
        loss.backward()
        ipu.outputs([input1.grad])

    input1.grad.zero_()
    input1.grad.detach_()
    out = F.log_softmax(input1, dim)
    loss = torch.sum(out)
    loss.backward()

    cpu_result = input1.grad
    ipu_result = ipu(input1)
    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result,
                            actual=cpu_result,
                            equal_nan=True)
