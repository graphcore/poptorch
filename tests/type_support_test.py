#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
import torch
import torch.nn as nn
import pytest
import helpers
import poptorch

MANY_TYPES = (torch.float16, torch.float32, torch.float64, torch.int32,
              torch.int64)

DEMOTED_ON_IPU = (torch.float64, torch.int64)


def get_simple_adder(return_type):
    class SimpleAdder(nn.Module):
        def forward(self, x, y):
            return (x + y).type(return_type)

    return poptorch.inferenceModel(SimpleAdder())


@pytest.mark.parametrize("input_type", MANY_TYPES)
@pytest.mark.parametrize("output_type", MANY_TYPES)
def test_many_input_output_types(input_type, output_type):
    model = get_simple_adder(output_type)
    t1 = torch.tensor([1.0, 25, -1.0, 83], dtype=input_type)
    t2 = torch.tensor([2.0, 35, 1.0, 32.4], dtype=input_type)

    helpers.set_device_ipu()
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
def test_many_implicit_cast(input_1_type, input_2_type, output_type):

    model = get_simple_adder(output_type)
    t1 = torch.tensor([1.0, 25., -1.0, 83.], dtype=input_1_type)
    t2 = torch.tensor([2.0, 35., 1.0, 32.4], dtype=input_2_type)

    helpers.assert_allclose(actual=model(t1, t2),
                            expected=torch.tensor([3., 60., 0., 115.4]),
                            atol=0.5,
                            rtol=0)


def get_unpack_clamp():
    class UnpackClamp(nn.Module):
        def forward(self, x):
            i, _ = x
            return i.clamp(-1, 1)

    return poptorch.inferenceModel(UnpackClamp())


@pytest.mark.parametrize("input_type", MANY_TYPES)
def test_clamp_many_types(input_type):
    model = get_unpack_clamp()
    x = torch.tensor([[-2, -1, 0, 1, 2], [0, 0, 0, 0, 0]], dtype=input_type)

    y = model(x)

    np.testing.assert_allclose(y.numpy(), np.array([-1, -1, 0, 1, 1]))


def get_simple_add_two():
    class GetSimpleAddTwo(nn.Module):
        def forward(self, x):
            return x + 2

    return poptorch.inferenceModel(GetSimpleAddTwo())


@pytest.mark.parametrize("input_type", MANY_TYPES)
def test_add_two_many_types(input_type):
    model = get_simple_add_two()

    t = torch.tensor([1.0, 25., -1.0, 83.], dtype=input_type)
    helpers.assert_allclose(actual=model(t),
                            expected=torch.tensor([3.0, 27., 1, 85.]),
                            atol=0.5,
                            rtol=0)


def get_simple_incrementer(constant_type, return_type):
    class SimpleIncrementer(nn.Module):
        def forward(self, x):
            return (x + torch.tensor(
                1, dtype=constant_type,
                device=helpers.get_device())).type(return_type)

    return poptorch.inferenceModel(SimpleIncrementer())


@pytest.mark.parametrize("input_type", MANY_TYPES)
@pytest.mark.parametrize("constant_type", MANY_TYPES)
@pytest.mark.parametrize("output_type", MANY_TYPES)
def test_many_constant_implicit_cast(input_type, constant_type, output_type):
    #Will not trace
    if constant_type == torch.float16:
        return

    helpers.set_device_ipu()
    model = get_simple_incrementer(constant_type, output_type)
    t = torch.tensor([1.0, 25., -1.0, 83.], dtype=input_type)

    helpers.assert_allclose(actual=model(t),
                            expected=torch.tensor([2.0, 26., 0, 84.]),
                            atol=0.5,
                            rtol=0)


@pytest.mark.parametrize("input_1_type", MANY_TYPES)
@pytest.mark.parametrize("input_2_type", MANY_TYPES)
def test_many_implicit_cast_greater_than(input_1_type, input_2_type):
    class GreaterThan(nn.Module):
        def forward(self, x, y):
            return x > y

    model = poptorch.inferenceModel(GreaterThan())

    t1 = torch.tensor([1, -1, 2.0, 550.4], dtype=input_1_type)
    t2 = torch.tensor([2.4, 2, 1.0, 32.4], dtype=input_2_type)

    helpers.assert_allequal(actual=model(t1, t2),
                            expected=torch.tensor([False, False, True, True]))


@pytest.mark.parametrize("input_type", MANY_TYPES)
def test_many_implicit_cast_greater_than_one(input_type):
    class GreaterThanOne(nn.Module):
        def forward(self, x):
            return x > 1

    model = poptorch.inferenceModel(GreaterThanOne())

    t = torch.tensor([2.5, -1, 2.0, 550.4], dtype=input_type)

    helpers.assert_allequal(actual=model(t),
                            expected=torch.tensor([True, False, True, True]))


@pytest.mark.parametrize("input_1_type", MANY_TYPES)
@pytest.mark.parametrize("input_2_type", MANY_TYPES)
def test_many_implicit_cast_equals(input_1_type, input_2_type):
    class Equals(nn.Module):
        def forward(self, x, y):
            return x == y

    model = poptorch.inferenceModel(Equals())

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
def test_many_implicit_cast_equals_one(input_type):
    class EqualsOne(nn.Module):
        def forward(self, x):
            return x == 1

    model = poptorch.inferenceModel(EqualsOne())

    t = torch.tensor([2.5, 1, 2.0, 550.4], dtype=input_type)

    helpers.assert_allequal(actual=model(t),
                            expected=torch.tensor([False, True, False, False]))


@pytest.mark.parametrize("input_1_type", MANY_TYPES)
@pytest.mark.parametrize("input_2_type", MANY_TYPES)
def test_many_implicit_cast_less_than(input_1_type, input_2_type):
    class LessThan(nn.Module):
        def forward(self, x, y):
            return x < y

    model = poptorch.inferenceModel(LessThan())

    t1 = torch.tensor([1, -1, 2.0, 550.4], dtype=input_1_type)
    t2 = torch.tensor([2.4, 2, 1.0, 32.4], dtype=input_2_type)

    helpers.assert_allequal(actual=model(t1, t2),
                            expected=torch.tensor([True, True, False, False]))


@pytest.mark.parametrize("input_type", MANY_TYPES)
def test_many_implicit_cast_less_than_one(input_type):
    class LessThanOne(nn.Module):
        def forward(self, x):
            return x < 1

    model = poptorch.inferenceModel(LessThanOne())

    t = torch.tensor([2.5, -1, 2.0, 550.4], dtype=input_type)

    helpers.assert_allequal(actual=model(t),
                            expected=torch.tensor([False, True, False, False]))


@pytest.mark.parametrize("input_type", MANY_TYPES)
def test_many_implicit_cast_one_less_than(input_type):
    class OneLessThan(nn.Module):
        def forward(self, x):
            return 1 < x  # pylint: disable=misplaced-comparison-constant

    model = poptorch.inferenceModel(OneLessThan())

    t = torch.tensor([2.5, -1, 2.0, 550.4], dtype=input_type)

    helpers.assert_allequal(actual=model(t),
                            expected=torch.tensor([True, False, True, True]))


@pytest.mark.parametrize("input_type", [torch.int8, torch.uint8, torch.int16])
def test_small_int(input_type):
    class Model(nn.Module):
        def forward(self, x):
            return x.float()

    input = torch.arange(100)

    # Convert to desired input type
    input = input.to(input_type)

    model = poptorch.inferenceModel(Model())

    output = model(input)

    assert output.dtype == torch.float
    helpers.assert_allequal(actual=output, expected=input.float())


@pytest.mark.parametrize("input_type", [torch.int8, torch.uint8, torch.int16])
def test_small_int_return(input_type):
    class Model(nn.Module):
        def forward(self, x):
            return x, x.float() + x.float()

    input = torch.arange(100)

    # Convert to desired input/output type
    input = input.to(input_type)

    model = poptorch.inferenceModel(Model())

    output, _ = model(input)

    assert output.dtype == input_type
    helpers.assert_allequal(actual=output, expected=input)


def test_tuple_and_list_constant():
    class Model(torch.nn.Module):
        def forward(self):
            const1 = torch.tensor([1., 2.], device=helpers.get_device())
            const2 = torch.tensor([3., 4.], device=helpers.get_device())

            return torch.tensor(1), const1 + const2, [const1, const2]

    model = Model()
    inference_model = poptorch.inferenceModel(model)

    helpers.set_device_ipu()
    poptorch_out = inference_model()

    helpers.set_device_cpu()
    native = model()
    helpers.assert_allclose(actual=poptorch_out, expected=native)


def test_tuple_and_list_constant_double_nested():
    class Model(torch.nn.Module):
        def forward(self):
            const1 = torch.tensor([1., 2.], device=helpers.get_device())
            const2 = torch.tensor([3., 4.], device=helpers.get_device())

            return ([torch.tensor(1)], const1 + const2,
                    ([const1, const2], [const1, const2]), const2)

    model = Model()
    inference_model = poptorch.inferenceModel(model)

    helpers.set_device_ipu()
    poptorch_out = inference_model()
    helpers.set_device_cpu()
    native = model()
    helpers.assert_allclose(actual=poptorch_out, expected=native)
