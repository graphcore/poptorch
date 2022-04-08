#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import torch.nn.functional as F
import pytest
import helpers
from poptorch.experimental import IPUContext


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
    F.relu, torch.tanh, torch.sigmoid, F.gelu, F.hardsigmoid, F.silu,
    F.hardswish, relu_inplace, tanh_inplace, sigmoid_inplace, silu_inplace,
    hardsigmoid_inplace, hardswish_inplace
]


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("op", activation_functions)
def test_activations(op):
    torch.manual_seed(42)

    input = torch.randn([2, 10, 4, 2])

    ipu_result = IPUContext(op)(input)
    cpu_result = op(input)

    assert ipu_result.size() == cpu_result.size()

    tol = [0.01, 1e-3] if op is F.gelu else [1e-4, 1e-7]

    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            atol=tol[0],
                            rtol=tol[1],
                            equal_nan=True)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_softmax(dim):
    torch.manual_seed(42)

    input1 = torch.randn([2, 2, 4, 5])

    cpu_result = F.softmax(input1, dim)
    ipu_result = IPUContext(F.softmax)(input1, dim)

    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            equal_nan=True)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_logsoftmax_forward(dim):
    torch.manual_seed(42)
    input1 = torch.randn([2, 2, 4, 5])

    cpu_result = F.log_softmax(input1, dim)
    ipu_result = IPUContext(F.log_softmax)(input1, dim)

    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            equal_nan=True)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_logsoftmax_backward(dim):
    torch.manual_seed(42)
    input1 = torch.nn.parameter.Parameter(torch.randn([2, 2, 4, 5]))

    def log_softmax_backward(t):
        out = F.log_softmax(t, dim)
        loss = torch.sum(out)
        loss.backward()
        return t.grad

    ipu_result = IPUContext(log_softmax_backward)(input1)
    input1.grad.zero_()
    input1.grad.detach_()

    cpu_result = log_softmax_backward(input1)

    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            equal_nan=True)
