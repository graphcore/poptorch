#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import helpers
from poptorch.experimental import IPUContext


def gt(t1, t2):
    return t1 > t2


def lt(t1, t2):
    return t1 < t2


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("op", [gt, lt])
def test_gt_lt(op):
    torch.manual_seed(42)

    t1 = torch.randn(3, 3)
    t2 = torch.randn(3, 3)

    cpu_result = op(t1, t2)
    ipu_result = IPUContext(op)(t1, t2)

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
