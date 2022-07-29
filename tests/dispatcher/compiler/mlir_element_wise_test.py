#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import ast
from copy import deepcopy
import os
import re
import pytest
import torch
import helpers

import poptorch
from poptorch.experimental import IPUContext


def op_harness(fn, *args, **kwargs):
    cpu_args = deepcopy(args)
    # Get the result from the CPU
    try:
        cpu_res = fn(*cpu_args, **kwargs)
    except RuntimeError as err:
        str_err = str(err)
        msg = f'({re.escape(str_err)}|Verification failed for)'

        cannot_cast = "can't be cast to the desired output type"
        if cannot_cast in str_err:
            msg = cannot_cast

        if "not implemented for" in str_err:
            msg = None  # as the error message is likely to be different anyway

        with pytest.raises((RuntimeError, poptorch.Error), match=msg):
            IPUContext(fn)(*args, **kwargs)

        return

    print(f"From CPU: {cpu_res}")

    ipu_res = IPUContext(fn)(*args, **kwargs)
    print(f"From IPU: {ipu_res}")

    check_dtype = isinstance(
        cpu_res, torch.Tensor
    ) and cpu_res.dtype != torch.int64 and cpu_res.dtype != torch.int32
    helpers.assert_allclose(expected=cpu_res,
                            actual=ipu_res,
                            check_dtype=check_dtype,
                            equal_nan=True)


binary_ops = [
    lambda x, y: x > y,
    lambda x, y: x >= y,
    lambda x, y: x < y,
    lambda x, y: x <= y,
    lambda x, y: x == y,
    lambda x, y: x != y,
    torch.logical_and,
    torch.logical_or,
    torch.logical_xor,
    lambda x, y: x & y,
    lambda x, y: x | y,
    lambda x, y: x ^ y,
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x / y,
    lambda x, y: x * y,
    torch.atan2,
    torch.max,
    torch.min,
    torch.minimum,
    torch.maximum,
    torch.remainder,
    torch.fmod,
    torch.floor_divide,
    torch.pow,
]

torch.manual_seed(42)
binary_test_cases = [
    (torch.tensor(2.0), torch.tensor(1.0)),
    (torch.tensor(70), torch.tensor(3)), (torch.tensor(2), torch.tensor(1.0)),
    (torch.tensor([]), torch.tensor([])),
    (torch.tensor([[1]],
                  dtype=torch.int32), torch.tensor([[5]], dtype=torch.int32)),
    (torch.tensor([[True, False],
                   [True, False]]), torch.tensor([[False, True], [True,
                                                                  False]])),
    (torch.tensor([12.0, 12.0, -12.0, -12.0, 11.0, 11.0, -11.0, -11.0]),
     torch.tensor([3.0, -3.0, 3.0, -3.0, 3.0, -3.0, 3.0, -3.0])),
    (torch.tensor([12, 12, -12, -12, 11, 11, -11,
                   -11]), torch.tensor([3, -3, 3, -3, 3, -3, 3, -3])),
    (torch.randn(3, 3), torch.randn(3, 3))
]


@pytest.mark.mlirSupportRequired
@pytest.mark.filterwarnings("ignore:floor_divide is deprecated")
@pytest.mark.parametrize("op", binary_ops)
@pytest.mark.parametrize("input", binary_test_cases)
def test_binary(op, input):
    op_harness(op, *input)


broadcast_test_shapes = [
    ((3, 1), (4)),
    ((3), (4, 3)),
    ((5, 1, 4, 1), (3, 1, 1)),
]


@pytest.mark.mlirSupportRequired
@pytest.mark.filterwarnings("ignore:floor_divide is deprecated")
@pytest.mark.parametrize("shapes", broadcast_test_shapes)
@pytest.mark.parametrize("op", binary_ops)
def test_binary_broadcast(op, shapes):
    input = tuple(map(torch.rand, shapes))
    op_harness(op, *input)


@pytest.mark.mlirSupportRequired
@pytest.mark.filterwarnings("ignore:floor_divide is deprecated")
@pytest.mark.parametrize("op", binary_ops)
def test_binary_nan_support(monkeypatch, op):
    poplar_engine_opts = os.getenv("POPLAR_ENGINE_OPTIONS")
    if poplar_engine_opts:
        poplar_engine_opts = ast.literal_eval(poplar_engine_opts)
    else:
        poplar_engine_opts = {}

    poplar_engine_opts["debug.floatPointOpException"] = "false"
    monkeypatch.setenv("POPLAR_ENGINE_OPTIONS",
                       str(poplar_engine_opts).replace('\'', '"'))

    lhs = torch.tensor([torch.nan, torch.inf, torch.nan, torch.inf])
    rhs = torch.tensor([torch.nan, torch.inf, 0.0, 0.0])

    if op is torch.max or op is torch.maximum:
        pytest.skip("TODO(T62888) max of nan and 0.0 should be nan it is "
                    "failing on the hardware")
    if op is torch.min or op is torch.minimum:
        pytest.skip("TODO(T62888) min of nan and 0.0 should be nan it is "
                    "failing on the model")

    op_harness(op, lhs, rhs)


unary_ops = [
    lambda x: -x,
    torch.abs,
    torch.acos,
    torch.acosh,
    torch.asin,
    torch.asinh,
    torch.atan,
    torch.atanh,
    torch.bitwise_not,
    torch.ceil,
    torch.cos,
    torch.cosh,
    torch.erf,
    torch.erfc,
    torch.exp,
    torch.floor,
    torch.frac,
    torch.isnan,
    torch.log,
    torch.log10,
    torch.log1p,
    torch.log2,
    torch.logical_not,
    torch.neg,
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

unary_test_cases = [
    torch.tensor(2.0),
    torch.tensor([]),
    torch.tensor([70, 0]),
    torch.tensor([70.25, 0.125]),
    torch.tensor([[13, 8]], dtype=torch.int32),
    torch.tensor([[True, False], [True, False]]),
    torch.randn(3, 3),
    torch.tensor([torch.nan, torch.inf, -torch.inf]),
]


@pytest.mark.mlirSupportRequired
@pytest.mark.filterwarnings("ignore:floor_divide is deprecated")
@pytest.mark.parametrize("op", unary_ops)
@pytest.mark.parametrize("input", unary_test_cases)
def test_unary(op, input):
    if op in (torch.asinh, torch.log1p) and torch.any(torch.isinf(input)):
        pytest.skip("TODO(T62888) `-inf` -> `nan`")

    torch.manual_seed(42)
    op_harness(op, input)


inplace_unary_ops = [
    lambda x: x.bitwise_not_(),
    # lambda x: x.logical_not_(),  # TODO(T66820): popops::logicalNotInPlace
    torch.abs_,
    torch.acos_,
    torch.acosh_,
    torch.asin_,
    torch.asinh_,
    torch.atan_,
    torch.atanh_,
    torch.ceil_,
    torch.cos_,
    torch.cosh_,
    torch.erf_,
    torch.erfc_,
    torch.exp_,
    torch.expm1_,
    torch.floor_,
    torch.frac_,
    torch.log_,
    torch.log10_,
    torch.log1p_,
    torch.log2_,
    torch.neg_,
    torch.reciprocal_,
    torch.round_,
    torch.rsqrt_,
    torch.sigmoid_,
    torch.sin_,
    torch.sinh_,
    torch.sqrt_,
    torch.tan_,
    torch.tanh_,
    torch.trunc_,
]


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("op", inplace_unary_ops)
@pytest.mark.parametrize("input", unary_test_cases)
def test_inplace_unary(op, input):
    if op in (torch.asinh_, torch.log1p_) and torch.any(torch.isinf(input)):
        pytest.skip("TODO(T62888) `-inf` -> `nan`")

    torch.manual_seed(42)
    op_harness(op, input)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("input", unary_test_cases)
@pytest.mark.parametrize("threshold", [0., 1000., -50])
@pytest.mark.parametrize("value", [0., 1000, -50])
def test_threshold(input, threshold, value):
    torch.manual_seed(42)
    op_harness(torch.threshold, input, threshold, value)


addc_test_cases = (
    ((2, 2), (2, 2), (2, 2)),
    ((3, 2, 2), (1, 2, 2), (3, 2, 1)),
    ((1, 1, 2), (4, 2), (2, 4, 1)),
)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("input_shapes", addc_test_cases)
@pytest.mark.parametrize("op", (torch.addcmul, torch.addcdiv))
def test_addc(op, input_shapes):
    inputs = [torch.randn(input_shape) for input_shape in input_shapes]
    op_harness(op, *inputs)


clamp_test_cases = [
    # TODO(pytorch 1.12): currently pytorch does not correctly call the tensor
    # overload when given a Tensor argument with empty shape
    #(torch.tensor(False)),
    #(torch.tensor(False), torch.tensor(True)),
    #(torch.tensor(1), torch.tensor(1)),
    (torch.tensor([0.0])),
    (torch.tensor([False]), torch.tensor([True]), torch.tensor([True])),
    (torch.tensor([2]), torch.tensor([3]), torch.tensor([1])),
    (torch.tensor([0.0, 1.0, 3.0]), torch.tensor([-1.0, 1.0, 5.0])),
    (torch.tensor([0.0, 1.0, 3.0]), None, torch.tensor([-1.0, 1.0, 5.0])),
    (torch.tensor([-2.0, -1.0, 1.0, 2.0]), torch.tensor([1.0, 1.0, 1.0, 1.0]),
     torch.tensor([-1.0, -1.0, -1.0, -1.0])),
    (torch.tensor([[0.0, 1.0, 3.0],
                   [0.0, 1.0,
                    3.0]]), torch.tensor([[5.0, 1.0, -1.0], [-1.0, 1.0, 5.0]]),
     torch.tensor([[-1.0, 1.0, 5.0], [-1.0, 1.0, 5.0]])),
    (torch.tensor([]), torch.tensor([])),
    (torch.tensor([]), torch.tensor([]), torch.tensor([])),
]


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("input", clamp_test_cases)
def test_clamp(input):
    op_harness(torch.clamp, *input)


clamp_broadcast_tests = (
    ((2, 2), (2, 2), (2, 2)),
    ((3, 2, 2), (1, 2, 2), (3, 2, 1)),
    ((1, 1, 2), (4, 2)),
)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("input_shapes", clamp_broadcast_tests)
def test_clamp_broadcast(input_shapes):
    inputs = [torch.randn(input_shape) for input_shape in input_shapes]
    op_harness(torch.clamp, *inputs)


MANY_TYPES = (torch.float16, torch.float32, torch.int32)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("input_1_type", MANY_TYPES)
@pytest.mark.parametrize("input_2_type", MANY_TYPES)
def test_many_implicit_casting(input_1_type, input_2_type):
    torch.manual_seed(42)

    t1 = torch.tensor([1.0, 25, -1.0, 83], dtype=input_1_type)
    t2 = torch.tensor([2.0, 35, 1.0, 32.4], dtype=input_2_type)

    class SimpleAdder(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = SimpleAdder()

    cpu_result = model(t1, t2)
    ipu_result = IPUContext(model)(t1, t2)

    helpers.assert_allequal(actual=ipu_result, expected=cpu_result)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("input_1_type", MANY_TYPES)
@pytest.mark.parametrize("input_2_type", MANY_TYPES)
@pytest.mark.parametrize("return_type", MANY_TYPES)
def test_many_implicit_casting_with_return(input_1_type, input_2_type,
                                           return_type):
    torch.manual_seed(42)

    t1 = torch.tensor([1.0, 25, -1.0, 83], dtype=input_1_type)
    t2 = torch.tensor([2.0, 35, 1.0, 32.4], dtype=input_2_type)

    class SimpleAdder(torch.nn.Module):
        def forward(self, x, y):
            return (x + y).type(return_type)

    model = SimpleAdder()

    cpu_result = model(t1, t2)
    ipu_result = IPUContext(model)(t1, t2)

    # Some cases including f16+f16 -> f32 differ in result.
    if input_1_type != torch.float16 and input_2_type != torch.float16:
        helpers.assert_allequal(actual=ipu_result, expected=cpu_result)
    else:
        helpers.assert_allclose(actual=ipu_result,
                                expected=cpu_result,
                                rtol=0.001,
                                atol=0.001)
