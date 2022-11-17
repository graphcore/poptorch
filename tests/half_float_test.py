#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import helpers
import poptorch


def assert_same_type(inputs, model):
    native_out = model(inputs)

    pop_model = poptorch.inferenceModel(model)
    pop_out = pop_model(inputs)

    assert native_out.dtype == pop_out.dtype


def type_out_harness(inputs, forward_op):
    class Model(torch.nn.Module):
        def forward(self, x):
            return forward_op(x)

    model = Model()

    assert_same_type(inputs, model)


## Ones and Zeros tests ##

ones_zeros = [torch.ones, torch.zeros]


# Tracing: input dtype always resolves to float32, because it is traced as float32,
# and the input itself is not used standalone.
#
# Dispatcher: The dtype will match what was requested.
@pytest.mark.parametrize("op", ones_zeros)
def test_ones_zeros_default_resolved(op):
    def fw_op(input):
        return op((2, 3, 4), dtype=input.dtype,
                  device=helpers.outputDevice()) + input.to(input.dtype)

    type_out_harness(torch.tensor([1], dtype=torch.float16), fw_op)
    type_out_harness(torch.tensor([1], dtype=torch.float32), fw_op)


# The dtype will correctly resolve becuse it matches the input added
# All settings will match pytorch
@pytest.mark.parametrize("op", ones_zeros)
def test_ones_zeros_input_resolved_with_input_dtype(op):
    def fw_op(input):
        return op((2, 3, 4), dtype=input.dtype,
                  device=helpers.outputDevice()) + input

    type_out_harness(torch.tensor([1], dtype=torch.float16), fw_op)
    type_out_harness(torch.tensor([1], dtype=torch.float32), fw_op)


# The zeros/ones will resolve correctly becaue torch.float16 could not have been
# from a tensor which could have beeh half/float.
#
# Half and half to float:
# The output will always be float 16.
#
# Like pytorch:
# The output will be correct.


@pytest.mark.parametrize("op", ones_zeros)
def test_ones_zeros_input_resolved_always_float16(op):
    def fw_op(input):
        return op(
            (2, 3, 4), dtype=torch.float16,
            device=helpers.outputDevice()) + input

    type_out_harness(torch.tensor([1], dtype=torch.float16), fw_op)
    type_out_harness(torch.tensor([1], dtype=torch.float32), fw_op)


# The dtype will resolve to the same as input. In the float16 case, the
# ones/zeros will be wrongly generated as a float16.
#
# The output will always match input.
@pytest.mark.parametrize("op", ones_zeros)
def test_ones_zeros_input_resolved_always_float32(op):
    def fw_op(input):
        return op(
            (2, 3, 4), dtype=torch.float32,
            device=helpers.outputDevice()) + input

    type_out_harness(torch.tensor([1], dtype=torch.float16), fw_op)
    type_out_harness(torch.tensor([1], dtype=torch.float32), fw_op)


## torch.rand tests ##


def test_rand_default_resolved():
    def fw_op(input):
        return torch.rand(3, 5, 100, dtype=input.dtype)

    type_out_harness(torch.tensor([1], dtype=torch.float16), fw_op)
    type_out_harness(torch.tensor([1], dtype=torch.float32), fw_op)


#The dtype will correctly resolve becuse it matches the input added
def test_rand_default_input_resolved():
    def fw_op(input):
        return torch.rand(3, 5, 100, dtype=input.dtype) + input

    type_out_harness(torch.tensor([1], dtype=torch.float16), fw_op)
    type_out_harness(torch.tensor([1], dtype=torch.float32), fw_op)


# The type will resolve correctly because torch.float16 could not have been
# from a tensor which could have been half/float.
#
# Half and half to float:
# The output will always be float 16.
#
# Like pytorch:
# The output will be correct.
def test_rand_default_input_resolved_always_float16():
    def fw_op(input):
        return torch.rand(3, 5, 100, dtype=torch.float16) + input

    type_out_harness(torch.tensor([1], dtype=torch.float16), fw_op)
    type_out_harness(torch.tensor([1], dtype=torch.float32), fw_op)


## torch.normal tests ##


# The type will be resolved correctly as the mean and standard deviation are
# inputs to the op
def test_normal_mean_correctly_resolved():
    def fw_op(input_mean):
        return torch.normal(input_mean, 10.0)

    type_out_harness(torch.tensor([0.0], dtype=torch.float16), fw_op)
    type_out_harness(torch.tensor([0.0], dtype=torch.float32), fw_op)


# The type will be resolved correctly as the mean and standard deviation are
# inputs to the op
def test_normal_std_correctly_resolved():
    def fw_op(input_std):
        return torch.normal(0.0, input_std)

    type_out_harness(torch.tensor([10.0], dtype=torch.float16), fw_op)
    type_out_harness(torch.tensor([10.0], dtype=torch.float32), fw_op)


## torch.distributions.uniform.Uniform tests ##


# The type will always resolve to float32 as it is traced to torch.rand without
# the low and high input tensors (which become dead code)
def test_distributions_uniform():
    def fw_op(input_low):
        torch.manual_seed(42)
        ud = torch.distributions.uniform.Uniform(
            input_low, torch.tensor([10.0], dtype=torch.float32))
        return ud.sample((10, 10, 1000))

    type_out_harness(torch.tensor([1], dtype=torch.float16), fw_op)
    type_out_harness(torch.tensor([1], dtype=torch.float32), fw_op)


## torch.distributions.Normal tests ##


# The type will resolve correctly because the mean is an input
def test_distributions_normal_mean_correctly_resolved():
    def fw_op(input_mean):
        torch.manual_seed(42)
        ud = torch.distributions.Normal(input_mean, 10.0)
        return ud.sample((10, 10, 100))

    type_out_harness(torch.tensor([0.0], dtype=torch.float16), fw_op)
    type_out_harness(torch.tensor([0.0], dtype=torch.float32), fw_op)


def test_distributions_normal_std_correctly_resolved():
    def fw_op(input_std):
        torch.manual_seed(42)
        ud = torch.distributions.Normal(0.0, input_std)
        return ud.sample((10, 10, 100))

    type_out_harness(torch.tensor([10.0], dtype=torch.float16), fw_op)
    type_out_harness(torch.tensor([10.0], dtype=torch.float32), fw_op)


## tensor._uniform test #


# The type will resolve correctly because it is based on the input tensor
def test_uniform_correctly_resolved():
    def fw_op(input_tensor):
        torch.manual_seed(42)
        input_tensor = input_tensor + 0  # Ensure input is not modified in place
        return input_tensor.uniform_()

    type_out_harness(torch.empty((3, 4, 10), dtype=torch.float16), fw_op)
    type_out_harness(torch.empty((3, 4, 10), dtype=torch.float32), fw_op)


## tensor._normal test #


# The type will also resolve correctly because it is based on the input tensor
def test_normal_correctly_resolved():
    def fw_op(input_tensor):
        torch.manual_seed(42)
        input_tensor = input_tensor + 0  # Ensure input is not modified in place
        return input_tensor.normal_()

    type_out_harness(torch.empty((3, 4, 10), dtype=torch.float16), fw_op)
    type_out_harness(torch.empty((3, 4, 10), dtype=torch.float32), fw_op)


## tensor constant tests ##


# The type will resolve correctly because it is added to the input.
#
# The output will always be the same as the
def test_constant_correctly_resolved():
    def fw_op(input):
        return torch.tensor([1, 2, 3], dtype=input.dtype) + input

    type_out_harness(torch.tensor([3, 4, 8], dtype=torch.float16), fw_op)
    type_out_harness(torch.tensor([3, 4, 8], dtype=torch.float32), fw_op)


# The type will resolve to float16 always because the input is cast to float16
# The output will always be float 16.
def test_constant_add_float16():
    def fw_op(input):
        return torch.tensor([1, 2, 3], dtype=input.dtype) + input.to(
            torch.float16)

    type_out_harness(torch.tensor([3, 4, 8], dtype=torch.float16), fw_op)
    type_out_harness(torch.tensor([3, 4, 8], dtype=torch.float32), fw_op)


# The type will resolve to the input rather than float32 because of the
# ambiguity betwen tracing with a float and a half converted to a float.
def test_constant_always_float32():
    def fw_op(input):
        return torch.tensor([1, 2, 3], dtype=torch.float32) + input

    type_out_harness(torch.tensor([3, 4, 8], dtype=torch.float16), fw_op)
    type_out_harness(torch.tensor([3, 4, 8], dtype=torch.float32), fw_op)


@pytest.mark.parametrize("conv", [True, False])
def test_float16_activations_float32_weights(conv):
    torch.manual_seed(42)

    if conv:
        input = torch.ones(1, 4, 4)
        model = torch.nn.Conv1d(4, 5, 2)
    else:
        input = torch.ones(10)
        model = torch.nn.Linear(10, 20)

    # Float 32 act, float 32 weights
    pop_model = poptorch.inferenceModel(model)
    pop_out = pop_model(input)

    assert pop_out.dtype == torch.float

    # Float 16 act, float 32 weights
    pop_model = poptorch.inferenceModel(model)
    pop_out = pop_model(input.half())
    assert pop_out.dtype == torch.half

    # Float 32 act, float 16 weights
    model.half()
    pop_model = poptorch.inferenceModel(model)
    pop_out = pop_model(input)
    assert pop_out.dtype == torch.float

    # Float 16 act, float 16 weights
    pop_model = poptorch.inferenceModel(model)
    pop_out = pop_model(input.half())
    assert pop_out.dtype == torch.half


def test_master_weight_training():
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.loss = torch.nn.MSELoss()

        def forward(self, data, target):
            out = self.linear(data)
            loss = self.loss(out, target)
            return out, loss

    model = Model()
    poptorch_model = poptorch.trainingModel(model)

    target = torch.randn(10)
    input = torch.randn(10).half()

    # Make sure the first run doesn't already pass the test.s
    original, original_loss = poptorch_model(input, target.half())
    assert original_loss > 0.1
    assert not torch.allclose(original.float(), target, rtol=1e-02, atol=1e-02)

    for _ in range(0, 2500):
        out, loss = poptorch_model(input, target.half())

    # Check we have trained the "model"
    assert loss.float() < 0.001
    helpers.assert_allclose(actual=out.float(),
                            expected=target,
                            rtol=1e-02,
                            atol=1e-02)


def test_bigger_model_training():
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_chain = torch.nn.Sequential(torch.nn.Linear(10, 10),
                                                    torch.nn.Linear(10, 10),
                                                    torch.nn.Linear(10, 10),
                                                    torch.nn.Linear(10, 10),
                                                    torch.nn.Linear(10, 10))
            self.loss = torch.nn.MSELoss()

        def forward(self, data, target):
            out = self.linear_chain(data)
            loss = self.loss(out, target)
            return out, loss

    model = Model()
    poptorch_model = poptorch.trainingModel(model)

    target = torch.randn(10)
    input = torch.randn(10).half()

    # Make sure the first run doesn't already pass the test.s
    original, original_loss = poptorch_model(input, target.half())
    assert original_loss > 0.1
    assert not torch.allclose(original.float(), target, rtol=1e-02, atol=1e-02)

    for _ in range(0, 2500):
        out, loss = poptorch_model(input, target.half())

    # Check we have trained the "model"
    assert loss.float() < 0.001
    helpers.assert_allclose(actual=out.float(),
                            expected=target,
                            rtol=1e-02,
                            atol=1e-02)
