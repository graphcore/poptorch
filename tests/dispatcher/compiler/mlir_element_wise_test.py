#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import helpers
from poptorch.experimental import IPUContext


def op_harness(fn, *args, **kwargs):
    # Get the result from the CPU
    cpu_res = fn(*args, **kwargs)
    print(f"From CPU: {cpu_res}")

    ipu_res = IPUContext(fn)(*args, **kwargs)
    print(f"From IPU: {ipu_res}")

    check_dtype = isinstance(
        cpu_res, torch.Tensor
    ) and cpu_res.dtype != torch.int64 and cpu_res.dtype != torch.int32
    return helpers.assert_allclose(expected=cpu_res,
                                   actual=ipu_res,
                                   check_dtype=check_dtype,
                                   equal_nan=True)


# TODO(T62028): Fix the failing binary operations
binary_ops = [
    lambda x, y: x > y,
    lambda x, y: x >= y,
    lambda x, y: x < y,
    lambda x, y: x <= y,
    lambda x, y: x == y,
    lambda x, y: x + y,
    lambda x, y: x - y,
    #lambda x, y: x / y,
    lambda x, y: x * y,
    #torch.atan2,
    #lambda x, y: x << y,
    #lambda x, y: x >> y,
    #lambda x, y: x and y,
    #lambda x, y: x or y,
    # max of nan and 0.0 should be nan it is failing on the hardware
    #torch.max,
    # min of nan and 0.0 should be nan but is in fact 0.0
    #torch.min,
    #torch.pow,
]

torch.manual_seed(42)
binary_test_cases = [
    (torch.tensor(2.0), torch.tensor(1.0)),
    (torch.tensor(70), torch.tensor(3)),
    # TODO(T59576): Casting isn't implemented yet
    #(torch.tensor(2), torch.tensor(1.0)),
    (torch.tensor([]), torch.tensor([1.0])),
    # TODO(T62028): This doesn't work with addition we should replace bool addition with AND
    #(torch.tensor([[True, False], [True, False]]),
    #torch.tensor([[False, True], [True, False]])),
    (torch.tensor([torch.nan, torch.inf, torch.nan,
                   torch.inf]), torch.tensor([torch.nan, torch.inf, 0.0,
                                              0.0])),
    (torch.randn(3, 3), torch.randn(3, 3))
]


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("op", binary_ops)
@pytest.mark.parametrize("input", binary_test_cases)
def test_binary(op, input):
    op_harness(op, *input)


bitwise_ops = [
    lambda x, y: x & y,
    lambda x, y: x | y,
    lambda x, y: x ^ y,
]

bitwise_test_cases = [
    (torch.tensor(70), torch.tensor(3)),
    (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32)),
    (torch.tensor([[True, False],
                   [True, False]]), torch.tensor([[False, True], [True,
                                                                  False]])),
    (torch.tensor([70, 123],
                  dtype=torch.uint8), torch.tensor([3, 7], dtype=torch.uint8)),
]


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("op", binary_ops)
@pytest.mark.parametrize("input", binary_test_cases)
def test_bitwise(op, input):
    op_harness(op, *input)


@pytest.mark.mlirSupportRequired
def test_isnan():
    t = torch.tensor([torch.nan, 1.0, torch.inf])

    cpu_result = torch.isnan(t)
    ipu_result = IPUContext(torch.isnan)(t)

    helpers.assert_allequal(actual=ipu_result, expected=cpu_result)


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
