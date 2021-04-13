#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
import torch
import torch.nn as nn
import pytest
import poptorch
import helpers

MANY_TYPES = (torch.float32, torch.float64, torch.int32, torch.int64)

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

    output = model(t1, t2)

    if output_type not in DEMOTED_ON_IPU:
        assert output[0].dtype == output_type
        assert output[1].dtype == output_type

    np.testing.assert_allclose(output.numpy(),
                               np.array([3., 60., 0., 115.4]),
                               atol=0.5)


@pytest.mark.parametrize("input_1_type", MANY_TYPES)
@pytest.mark.parametrize("input_2_type", MANY_TYPES)
@pytest.mark.parametrize("output_type", MANY_TYPES)
def test_many_implicit_cast(input_1_type, input_2_type, output_type):

    model = get_simple_adder(output_type)
    t1 = torch.tensor([1.0, 25., -1.0, 83.], dtype=input_1_type)
    t2 = torch.tensor([2.0, 35., 1.0, 32.4], dtype=input_2_type)

    np.testing.assert_allclose(model(t1, t2).numpy(),
                               np.array([3., 60., 0., 115.4]),
                               atol=0.5)


def get_simple_add_two():
    class GetSimpleAddTwo(nn.Module):
        def forward(self, x):
            return x + 2

    return poptorch.inferenceModel(GetSimpleAddTwo())


@pytest.mark.parametrize("input_type", MANY_TYPES)
def test_add_two_many_types(input_type):
    model = get_simple_add_two()

    t = torch.tensor([1.0, 25., -1.0, 83.], dtype=input_type)
    np.testing.assert_allclose(model(t).numpy(),
                               np.array([3.0, 27., 1, 85.]),
                               atol=0.5)


def get_simple_incrementer(constant_type, return_type):
    class SimpleIncrementer(nn.Module):
        def forward(self, x):
            return (x + torch.tensor(1, dtype=constant_type)).type(return_type)

    return poptorch.inferenceModel(SimpleIncrementer())


@pytest.mark.parametrize("input_type", MANY_TYPES)
@pytest.mark.parametrize("constant_type", MANY_TYPES)
@pytest.mark.parametrize("output_type", MANY_TYPES)
def test_many_constant_implicit_cast(input_type, constant_type, output_type):
    #Will not trace
    if constant_type == torch.float16:
        return

    model = get_simple_incrementer(constant_type, output_type)
    t = torch.tensor([1.0, 25., -1.0, 83.], dtype=input_type)

    np.testing.assert_allclose(model(t).numpy(),
                               np.array([2.0, 26., 0, 84.]),
                               atol=0.5)


@pytest.mark.parametrize("input_1_type", MANY_TYPES)
@pytest.mark.parametrize("input_2_type", MANY_TYPES)
def test_many_implicit_cast_greater_than(input_1_type, input_2_type):
    class GreaterThan(nn.Module):
        def forward(self, x, y):
            return x > y

    model = poptorch.inferenceModel(GreaterThan())

    t1 = torch.tensor([1, -1, 2.0, 550.4], dtype=input_1_type)
    t2 = torch.tensor([2.4, 2, 1.0, 32.4], dtype=input_2_type)

    np.testing.assert_equal(
        model(t1, t2).numpy(), np.array([False, False, True, True]))


@pytest.mark.parametrize("input_type", MANY_TYPES)
def test_many_implicit_cast_greater_than_one(input_type):
    class GreaterThanOne(nn.Module):
        def forward(self, x):
            return x > 1

    model = poptorch.inferenceModel(GreaterThanOne())

    t = torch.tensor([2.5, -1, 2.0, 550.4], dtype=input_type)

    np.testing.assert_equal(
        model(t).numpy(), np.array([True, False, True, True]))


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

    if (input_1_type in (torch.float32, torch.float64)
            and input_2_type in (torch.float32, torch.float64)):
        depends = True

    if (input_1_type in (torch.int32, torch.int64)
            and input_2_type in (torch.int32, torch.int64)):
        depends = True

    assert np.all(
        model(t1, t2).numpy() == np.array([False, False, True, depends]))


@pytest.mark.parametrize("input_type", MANY_TYPES)
def test_many_implicit_cast_equals_one(input_type):
    class EqualsOne(nn.Module):
        def forward(self, x):
            return x == 1

    model = poptorch.inferenceModel(EqualsOne())

    t = torch.tensor([2.5, 1, 2.0, 550.4], dtype=input_type)

    np.testing.assert_equal(
        model(t).numpy(), np.array([False, True, False, False]))


@pytest.mark.parametrize("input_1_type", MANY_TYPES)
@pytest.mark.parametrize("input_2_type", MANY_TYPES)
def test_many_implicit_cast_less_than(input_1_type, input_2_type):
    class LessThan(nn.Module):
        def forward(self, x, y):
            return x < y

    model = poptorch.inferenceModel(LessThan())

    t1 = torch.tensor([1, -1, 2.0, 550.4], dtype=input_1_type)
    t2 = torch.tensor([2.4, 2, 1.0, 32.4], dtype=input_2_type)

    np.testing.assert_equal(
        model(t1, t2).numpy(), np.array([True, True, False, False]))


@pytest.mark.parametrize("input_type", MANY_TYPES)
def test_many_implicit_cast_less_than_one(input_type):
    class LessThanOne(nn.Module):
        def forward(self, x):
            return x < 1

    model = poptorch.inferenceModel(LessThanOne())

    t = torch.tensor([2.5, -1, 2.0, 550.4], dtype=input_type)

    np.testing.assert_equal(
        model(t).numpy(), np.array([False, True, False, False]))


@pytest.mark.parametrize("input_type", MANY_TYPES)
def test_many_implicit_cast_one_less_than(input_type):
    class OneLessThan(nn.Module):
        def forward(self, x):
            return 1 < x  # pylint: disable=misplaced-comparison-constant

    model = poptorch.inferenceModel(OneLessThan())

    t = torch.tensor([2.5, -1, 2.0, 550.4], dtype=input_type)

    np.testing.assert_equal(
        model(t).numpy(), np.array([True, False, True, True]))


@pytest.mark.parametrize("input_type", [torch.int8, torch.uint8])
def test_int8(input_type):
    class Model(nn.Module):
        def forward(self, x):
            return x.float()

    input = torch.arange(100)

    # Convert to int8/uint8
    input = input.to(input_type)

    model = poptorch.inferenceModel(Model())

    output = model(input)

    assert output.dtype == torch.float
    helpers.assert_allequal(actual=output, expected=input.float())


@pytest.mark.parametrize("input_type", [torch.int8, torch.uint8])
def test_int8_return(input_type):
    class Model(nn.Module):
        def forward(self, x):
            return x, x.float() + x.float()

    input = torch.arange(100)

    # Convert to int8/uint8
    input = input.to(input_type)

    model = poptorch.inferenceModel(Model())

    output, _ = model(input)

    assert output.dtype == input_type
    helpers.assert_allequal(actual=output, expected=input)
