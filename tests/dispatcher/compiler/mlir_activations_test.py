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


def leaky_relu_inplace(x):
    F.leaky_relu(x, inplace=True)
    return x


activation_functions = [
    F.elu,
    F.gelu,
    F.hardshrink,
    F.hardsigmoid,
    F.hardswish,
    F.leaky_relu,
    F.logsigmoid,
    F.relu,
    F.silu,
    F.softplus,
    F.softshrink,
    hardsigmoid_inplace,
    hardswish_inplace,
    leaky_relu_inplace,
    relu_inplace,
    sigmoid_inplace,
    silu_inplace,
    tanh_inplace,
    torch.sigmoid,
    torch.tanh,
]

activation_backward_functions = [torch.tanh, torch.sigmoid, F.leaky_relu]


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("op", activation_functions)
def test_activations(op):
    torch.manual_seed(42)

    input = torch.randn([2, 10, 4, 2])

    def training_step(x):
        if op in activation_backward_functions:
            x.requires_grad = True
            x.retain_grad()

        out = op(x)
        loss = torch.sum(out)
        loss.backward()
        return out, x.grad

    if op in activation_backward_functions:
        cpu_step = training_step
    else:
        cpu_step = op

    ipu_result = IPUContext(cpu_step)(input)
    input.grad = None

    cpu_result = cpu_step(input)

    tol = [0.01, 1e-3] if op is F.gelu else [1e-4, 1e-7]

    for cpu, ipu in zip(cpu_result, ipu_result):
        helpers.assert_allclose(expected=cpu,
                                actual=ipu,
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
    input1 = torch.randn([2, 2, 4, 5])

    def log_softmax_backward(t):
        t.requires_grad = True
        t.retain_grad()
        out = F.log_softmax(t, dim)
        loss = torch.sum(out)
        loss.backward()
        return t.grad

    ipu_result = IPUContext(log_softmax_backward)(input1)
    cpu_result = log_softmax_backward(input1)

    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            equal_nan=True)


@pytest.mark.mlirSupportRequired
def test_prelu():
    torch.manual_seed(42)

    inp = torch.randn((2, 2))
    weight = torch.zeros(2)

    cpu_result = F.prelu(inp, weight)
    ipu_result = IPUContext(F.prelu)(inp, weight)

    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            equal_nan=True)
