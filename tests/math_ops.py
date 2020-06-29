#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import poptorch

import pytest

# Unsupported and uncatergorised math ops.
# torch.addcdiv, torch.addcmul, torch.clamp, torch.lerp,
# torch.mvlgamma, torch.polygamma,


def unary_op_harness(op, input, eq):
    class Model(torch.nn.Module):
        def __init__(self, op):
            super(Model, self).__init__()
            self.op = op

        def forward(self, x):
            return self.op(x)

    model = Model(op)

    # Run on CPU.
    nativeOut = model(input)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input)

    assert eq(nativeOut, poptorch_out)


def binary_op_harness(model, input1, input2, eq):
    # Run on CPU.
    nativeOut = model(input1, input2)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input1, input2)

    assert eq(nativeOut, poptorch_out)


unary_ops_float = [
    torch.abs,
    # torch.acos,
    torch.asin,
    torch.atan,
    # torch.angle,
    torch.ceil,
    torch.cos,
    torch.cosh,
    # torch.conj, torch.digamma, torch.erf, torch.erfc, torch.erfinv
    torch.exp,
    torch.expm1,
    torch.floor,
    torch.frac,

    # torch.imag, torch.lgamma,
    torch.log,
    torch.log10,
    torch.log1p,
    torch.log2,
    # torch.logical_not, torch.mvlgamma,
    torch.neg,
    # torch.real,
    torch.reciprocal,
    torch.round,
    torch.rsqrt,
    torch.sigmoid,
    torch.sign,
    torch.sin,
    torch.sinh,
    torch.sqrt,
    torch.square,
    torch.tan,
    torch.tanh,
    torch.trunc,
]


@pytest.mark.parametrize("op", unary_ops_float)
def test_unary_ops_float(op):
    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 200])

    def compare(x, y):
        return torch.allclose(x, y, atol=1e-05, equal_nan=True)

    unary_op_harness(op, input, compare)


unary_ops_int = [  # torch.bitwise_not,
]


def test_binary_pow():
    torch.manual_seed(42)
    input1 = torch.randn([1, 2, 10, 200])

    def compare(x, y):
        return torch.allclose(x, y, atol=1e-05, equal_nan=True)

    def op(x):
        return torch.pow(x, 4.0)

    unary_op_harness(op, input1, compare)

    # Test "int" parameter
    def op(x):
        return torch.pow(x, 3)

    unary_op_harness(op, input1, compare)

    def op(x):
        return torch.pow(x, 2.5)

    unary_op_harness(op, input1, compare)


@pytest.mark.parametrize("op", unary_ops_int)
def test_unary_ops_int(op):
    torch.manual_seed(42)

    input = torch.randint(-1000, 1000, [1, 2, 10, 200])

    def compare(x, y):
        return torch.eq(x, y)

    unary_op_harness(op, input, compare)


binary_ops_float = [
    torch.add,
    # torch.atan2,
    torch.div,
    torch.sub,
    # torch.fmod,
    # torch.floor_divide,
    torch.mul,
    # torch.remainder,
    torch.true_divide
]


@pytest.mark.parametrize("op", binary_ops_float)
def test_binary_ops_float(op):
    torch.manual_seed(42)
    input1 = torch.randn([1, 2, 10, 200])
    input2 = torch.randn([1, 2, 10, 200])

    def compare(x, y):
        return torch.allclose(x, y, atol=1e-05, equal_nan=True)

    class Model(torch.nn.Module):
        def __init__(self, op):
            super(Model, self).__init__()
            self.op = op

        def forward(self, x, y):
            return self.op(x, y)

    binary_op_harness(Model(op), input1, input2, compare)


binary_ops_basic_element_wise_float = [
    torch.add,
    torch.div,
    torch.sub,
    torch.mul,
]


@pytest.mark.parametrize("op", binary_ops_basic_element_wise_float)
def test_binary_ops_elementwise_edgecases(op):
    torch.manual_seed(42)
    input1 = torch.randn([1, 2, 10, 200])
    input2 = torch.randn([1])

    def compare(x, y):
        return torch.allclose(x, y, atol=1e-05, equal_nan=True)

    # Constant on LHS
    class ConstOnLHS(torch.nn.Module):
        def __init__(self, op):
            super(ConstOnLHS, self).__init__()
            self.op = op

        def forward(self, x, y):
            return self.op(x, 4.0)

    binary_op_harness(ConstOnLHS(op), input1, input2, compare)

    # Constant on RHS
    class ConstOnRHS(torch.nn.Module):
        def __init__(self, op):
            super(ConstOnRHS, self).__init__()
            self.op = op

        def forward(self, x, y):
            return self.op(2.5, x)

    binary_op_harness(ConstOnRHS(op), input1, input2, compare)

    # Constant on LHS wrong type.
    class ConstOnLHSInt(torch.nn.Module):
        def __init__(self, op):
            super(ConstOnLHSInt, self).__init__()
            self.op = op

        def forward(self, x, y):
            return self.op(x, 4)

    binary_op_harness(ConstOnLHSInt(op), input1, input2, compare)

    # Constant on RHS wrong type
    class ConstOnRHSInt(torch.nn.Module):
        def __init__(self, op):
            super(ConstOnRHSInt, self).__init__()
            self.op = op

        def forward(self, x, y):
            return self.op(134, x)

    binary_op_harness(ConstOnRHSInt(op), input1, input2, compare)


binary_op_int = [
    # torch.logical_or, torch.logical_xor, , torch.bitwise_and, torch.bitwise_or, torch.bitwise_xor, torch.logical_and,
]

# These functions support API 1 - op(input)
reduction_ops_api1 = [
    torch.argmax,
    torch.argmin,
    # torch.dist,
    torch.mean,
    #torch.median, torch.mode, torch.norm,
    torch.prod,
    #torch.std, torch.std_mean,
    torch.sum,
    #torch.unique, torch.unique_consecutive,torch.var, torch.var_mean,
]

# These functions support API 2 - op(input,dim,keep_dim)
reduction_ops_api2 = [
    torch.argmax,
    torch.argmin,
    # torch.dist,
    torch.mean,
    #torch.median, torch.mode, torch.norm,
    torch.prod,
    torch.logsumexp,  # logsumexp doesn't support API 1.
    #torch.std, torch.std_mean,
    torch.sum,
    #torch.unique, torch.unique_consecutive,torch.var, torch.var_mean,
]


@pytest.mark.parametrize("op", reduction_ops_api1)
def test_reduction_ops_float(op):
    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 200])

    def compare(x, y):

        if x.dtype == torch.float32:
            return torch.allclose(x, y)
        else:
            return torch.eq(x, y)

    unary_op_harness(op, input, compare)


@pytest.mark.parametrize("op", reduction_ops_api2)
def test_reduction_ops_float_api2(op):
    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 200])

    def operation(x):
        return op(x, dim=2, keepdim=False)

    def compare(x, y):

        if x.dtype == torch.float32:
            return torch.allclose(x, y)
        else:

            if torch.numel(x) > 1:
                # Work around not returning longs from popart.
                return torch.equal(x, y.type_as(x))
            return torch.eq(x, y)

    unary_op_harness(operation, input, compare)


comparison_ops = [
    torch.allclose, torch.argsort, torch.eq, torch.equal, torch.ge, torch.gt,
    torch.isfinite, torch.isinf, torch.isnan, torch.kthvalue, torch.le,
    torch.lt, torch.max, torch.min, torch.ne, torch.sort, torch.topk
]
