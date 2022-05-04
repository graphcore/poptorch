#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
import torch
import torch.nn as nn
import pytest
import poptorch
import helpers

MANY_TYPES = (torch.float16, torch.float32, torch.float64, torch.int32,
              torch.int64)

DEMOTED_ON_IPU = (torch.float64, torch.int64)


def get_simple_adder(return_type, trace_model):
    class SimpleAdder(nn.Module):
        def forward(self, x, y):
            return (x + y).type(return_type)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    return poptorch.inferenceModel(SimpleAdder(), options)


@pytest.mark.parametrize("input_type", MANY_TYPES)
@pytest.mark.parametrize("output_type", MANY_TYPES)
@pytest.mark.parametrize("trace_model", [True, False])
def test_many_input_output_types(input_type, output_type, trace_model):
    model = get_simple_adder(output_type, trace_model)
    t1 = torch.tensor([1.0, 25, -1.0, 83], dtype=input_type)
    t2 = torch.tensor([2.0, 35, 1.0, 32.4], dtype=input_type)

    output = model(t1, t2)

    if output_type not in DEMOTED_ON_IPU:
        assert output[0].dtype == output_type
        assert output[1].dtype == output_type

    helpers.assert_allclose(actual=output,
                            expected=torch.tensor([3., 60., 0., 115.4]),
                            atol=0.5,
                            rtol=0)


@pytest.mark.parametrize("input_1_type", MANY_TYPES)
@pytest.mark.parametrize("input_2_type", MANY_TYPES)
@pytest.mark.parametrize("output_type", MANY_TYPES)
@pytest.mark.parametrize("trace_model", [True, False])
def test_many_implicit_cast(input_1_type, input_2_type, output_type,
                            trace_model):

    model = get_simple_adder(output_type, trace_model)
    t1 = torch.tensor([1.0, 25., -1.0, 83.], dtype=input_1_type)
    t2 = torch.tensor([2.0, 35., 1.0, 32.4], dtype=input_2_type)

    helpers.assert_allclose(actual=model(t1, t2),
                            expected=torch.tensor([3., 60., 0., 115.4]),
                            atol=0.5,
                            rtol=0)


def get_unpack_clamp(trace_model):
    class UnpackClamp(nn.Module):
        def forward(self, x):
            i, _ = x
            return i.clamp(-1, 1)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    return poptorch.inferenceModel(UnpackClamp(), options)


@pytest.mark.parametrize("input_type", MANY_TYPES)
@pytest.mark.parametrize("trace_model", [True, False])
def test_clamp_many_types(input_type, trace_model):
    model = get_unpack_clamp(trace_model)
    x = torch.tensor([[-2, -1, 0, 1, 2], [0, 0, 0, 0, 0]], dtype=input_type)

    y = model(x)

    np.testing.assert_allclose(y.numpy(), np.array([-1, -1, 0, 1, 1]))


def get_simple_add_two(trace_model):
    class GetSimpleAddTwo(nn.Module):
        def forward(self, x):
            return x + 2

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    return poptorch.inferenceModel(GetSimpleAddTwo(), options)


@pytest.mark.parametrize("input_type", MANY_TYPES)
@pytest.mark.parametrize("trace_model", [True, False])
def test_add_two_many_types(input_type, trace_model):
    model = get_simple_add_two(trace_model)

    t = torch.tensor([1.0, 25., -1.0, 83.], dtype=input_type)
    helpers.assert_allclose(actual=model(t),
                            expected=torch.tensor([3.0, 27., 1, 85.]),
                            atol=0.5,
                            rtol=0)


def get_simple_incrementer(constant_type, return_type, trace_model):
    class SimpleIncrementer(nn.Module):
        def forward(self, x):
            return (x + torch.tensor(1, dtype=constant_type)).type(return_type)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    return poptorch.inferenceModel(SimpleIncrementer(), options)


@pytest.mark.parametrize("input_type", MANY_TYPES)
@pytest.mark.parametrize("constant_type", MANY_TYPES)
@pytest.mark.parametrize("output_type", MANY_TYPES)
@pytest.mark.parametrize("trace_model", [True, False])
def test_many_constant_implicit_cast(input_type, constant_type, output_type,
                                     trace_model):
    #Will not trace
    if constant_type == torch.float16:
        return

    model = get_simple_incrementer(constant_type, output_type, trace_model)
    t = torch.tensor([1.0, 25., -1.0, 83.], dtype=input_type)

    helpers.assert_allclose(actual=model(t),
                            expected=torch.tensor([2.0, 26., 0, 84.]),
                            atol=0.5,
                            rtol=0)


@pytest.mark.parametrize("input_1_type", MANY_TYPES)
@pytest.mark.parametrize("input_2_type", MANY_TYPES)
@pytest.mark.parametrize("trace_model", [True, False])
def test_many_implicit_cast_greater_than(input_1_type, input_2_type,
                                         trace_model):
    class GreaterThan(nn.Module):
        def forward(self, x, y):
            return x > y

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    model = poptorch.inferenceModel(GreaterThan(), options)

    t1 = torch.tensor([1, -1, 2.0, 550.4], dtype=input_1_type)
    t2 = torch.tensor([2.4, 2, 1.0, 32.4], dtype=input_2_type)

    helpers.assert_allequal(actual=model(t1, t2),
                            expected=torch.tensor([False, False, True, True]))


@pytest.mark.parametrize("input_type", MANY_TYPES)
@pytest.mark.parametrize("trace_model", [True, False])
def test_many_implicit_cast_greater_than_one(input_type, trace_model):
    class GreaterThanOne(nn.Module):
        def forward(self, x):
            return x > 1

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    model = poptorch.inferenceModel(GreaterThanOne(), options)

    t = torch.tensor([2.5, -1, 2.0, 550.4], dtype=input_type)

    helpers.assert_allequal(actual=model(t),
                            expected=torch.tensor([True, False, True, True]))


@pytest.mark.parametrize("input_1_type", MANY_TYPES)
@pytest.mark.parametrize("input_2_type", MANY_TYPES)
@pytest.mark.parametrize("trace_model", [True, False])
def test_many_implicit_cast_equals(input_1_type, input_2_type, trace_model):
    class Equals(nn.Module):
        def forward(self, x, y):
            return x == y

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    model = poptorch.inferenceModel(Equals(), options)

    t1 = torch.tensor([1, -1, 2.0, 550.4], dtype=input_1_type)
    t2 = torch.tensor([2.4, 2, 2.0, 550.4], dtype=input_2_type)

    depends = False

    if (input_1_type == torch.float16 and input_2_type == torch.float16):
        depends = True

    if (input_1_type in (torch.float32, torch.float64)
            and input_2_type in (torch.float32, torch.float64)):
        depends = True

    if (input_1_type in (torch.int32, torch.int64)
            and input_2_type in (torch.int32, torch.int64)):
        depends = True

    helpers.assert_allequal(actual=model(t1, t2),
                            expected=torch.tensor(
                                [False, False, True, depends]))


@pytest.mark.parametrize("input_type", MANY_TYPES)
@pytest.mark.parametrize("trace_model", [True, False])
def test_many_implicit_cast_equals_one(input_type, trace_model):
    class EqualsOne(nn.Module):
        def forward(self, x):
            return x == 1

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    model = poptorch.inferenceModel(EqualsOne(), options)

    t = torch.tensor([2.5, 1, 2.0, 550.4], dtype=input_type)

    helpers.assert_allequal(actual=model(t),
                            expected=torch.tensor([False, True, False, False]))


@pytest.mark.parametrize("input_1_type", MANY_TYPES)
@pytest.mark.parametrize("input_2_type", MANY_TYPES)
@pytest.mark.parametrize("trace_model", [True, False])
def test_many_implicit_cast_less_than(input_1_type, input_2_type, trace_model):
    class LessThan(nn.Module):
        def forward(self, x, y):
            return x < y

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    model = poptorch.inferenceModel(LessThan(), options)

    t1 = torch.tensor([1, -1, 2.0, 550.4], dtype=input_1_type)
    t2 = torch.tensor([2.4, 2, 1.0, 32.4], dtype=input_2_type)

    helpers.assert_allequal(actual=model(t1, t2),
                            expected=torch.tensor([True, True, False, False]))


@pytest.mark.parametrize("input_type", MANY_TYPES)
@pytest.mark.parametrize("trace_model", [True, False])
def test_many_implicit_cast_less_than_one(input_type, trace_model):
    class LessThanOne(nn.Module):
        def forward(self, x):
            return x < 1

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    model = poptorch.inferenceModel(LessThanOne(), options)

    t = torch.tensor([2.5, -1, 2.0, 550.4], dtype=input_type)

    helpers.assert_allequal(actual=model(t),
                            expected=torch.tensor([False, True, False, False]))


@pytest.mark.parametrize("input_type", MANY_TYPES)
@pytest.mark.parametrize("trace_model", [True, False])
def test_many_implicit_cast_one_less_than(input_type, trace_model):
    class OneLessThan(nn.Module):
        def forward(self, x):
            return 1 < x  # pylint: disable=misplaced-comparison-constant

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    model = poptorch.inferenceModel(OneLessThan(), options)

    t = torch.tensor([2.5, -1, 2.0, 550.4], dtype=input_type)

    helpers.assert_allequal(actual=model(t),
                            expected=torch.tensor([True, False, True, True]))


@pytest.mark.parametrize("input_type", [torch.int8, torch.uint8])
@pytest.mark.parametrize("trace_model", [True, False])
def test_int8(input_type, trace_model):
    class Model(nn.Module):
        def forward(self, x):
            return x.float()

    input = torch.arange(100)

    # Convert to int8/uint8
    input = input.to(input_type)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    model = poptorch.inferenceModel(Model(), options)

    output = model(input)

    assert output.dtype == torch.float
    helpers.assert_allequal(actual=output, expected=input.float())


@pytest.mark.parametrize("input_type", [torch.int8, torch.uint8])
@pytest.mark.parametrize("trace_model", [True, False])
def test_int8_return(input_type, trace_model):
    class Model(nn.Module):
        def forward(self, x):
            return x, x.float() + x.float()

    input = torch.arange(100)

    # Convert to int8/uint8
    input = input.to(input_type)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    model = poptorch.inferenceModel(Model(), options)

    output, _ = model(input)

    assert output.dtype == input_type
    helpers.assert_allequal(actual=output, expected=input)


@pytest.mark.parametrize("trace_model", [True, False])
def test_tuple_and_list_constant(trace_model):
    const1 = torch.tensor([1., 2.])
    const2 = torch.tensor([3., 4.])

    class Model(torch.nn.Module):
        def forward(self):
            return torch.tensor(1), const1 + const2, [const1, const2]

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)

    poptorch_out = inference_model()
    native = model()
    helpers.assert_allclose(actual=poptorch_out, expected=native)


@pytest.mark.parametrize("trace_model", [True, False])
def test_tuple_and_list_constant_double_nested(trace_model):
    const1 = torch.tensor([1., 2.])
    const2 = torch.tensor([3., 4.])

    class Model(torch.nn.Module):
        def forward(self):
            return ([torch.tensor(1)], const1 + const2,
                    ([const1, const2], [const1, const2]), const2)

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)

    poptorch_out = inference_model()
    native = model()
    helpers.assert_allclose(actual=poptorch_out, expected=native)
