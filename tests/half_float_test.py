#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import poptorch
import pytest
import torch
import helpers


def type_out_harness(inputs, forward_op, expect_same_type):
    class Model(torch.nn.Module):
        def forward(self, x):
            return forward_op(x)

    model = Model()
    native_out = model(inputs)

    opts = poptorch.Options()
    pop_model = poptorch.inferenceModel(model, opts)
    pop_out = pop_model(inputs)

    if expect_same_type:
        assert native_out.dtype == pop_out.dtype
    else:
        assert native_out.dtype != pop_out.dtype


## Ones and Zeros tests ##

# The output type will always be torch.float32 as the tracing is always
# performed with a float32 and the input is not linked to the op
ones_zeros = [torch.ones, torch.zeros]


@pytest.mark.parametrize("op", ones_zeros)
def test_ones_zeros_default_resolved(op):
    def fw_op(input):
        return op((2, 3, 4), dtype=input.dtype) + input.to(input.dtype)

    type_out_harness(torch.tensor([1], dtype=torch.float16), fw_op, False)
    type_out_harness(torch.tensor([1], dtype=torch.float32), fw_op, True)


#The dtype will correctly resolve becuse it matches the input added
@pytest.mark.parametrize("op", ones_zeros)
def test_ones_zeros_input_resolved_with_input_dtype(op):
    def fw_op(input):
        return op((2, 3, 4), dtype=input.dtype) + input

    type_out_harness(torch.tensor([1], dtype=torch.float16), fw_op, True)
    type_out_harness(torch.tensor([1], dtype=torch.float32), fw_op, True)


#TODO(T29278) FIX
# The dtype will resolve to the same as input. In the float32 case, the
# ones/zeros will be wrongly generated as a float32, however the output type
# will remain the same
@pytest.mark.parametrize("op", ones_zeros)
def test_ones_zeros_input_resolved_always_float16(op):
    def fw_op(input):
        return op((2, 3, 4), dtype=torch.float16) + input

    type_out_harness(torch.tensor([1], dtype=torch.float32), fw_op, False)
    type_out_harness(torch.tensor([1], dtype=torch.float16), fw_op, True)


# The dtype will resolve to the same as input. In the float16 case, the
# ones/zeros will be wrongly generated as a float16, so the output type
# will be wrong
@pytest.mark.parametrize("op", ones_zeros)
def test_ones_zeros_input_resolved_always_float32(op):
    def fw_op(input):
        return op((2, 3, 4), dtype=torch.float32) + input

    type_out_harness(torch.tensor([1], dtype=torch.float32), fw_op, True)
    type_out_harness(torch.tensor([1], dtype=torch.float16), fw_op, False)


## torch.rand tests ##


# The dtype will always resolve to float32 as tracing always happens with
# float 32 and the input is not captured
def test_rand_default_resolved():
    def fw_op(input):
        return torch.rand(3, 5, 100, dtype=input.dtype)

    type_out_harness(torch.tensor([1], dtype=torch.float16), fw_op, False)
    type_out_harness(torch.tensor([1], dtype=torch.float32), fw_op, True)


#The dtype will correctly resolve becuse it matches the input added
def test_rand_default_input_resolved():
    def fw_op(input):
        return torch.rand(3, 5, 100, dtype=input.dtype) + input

    type_out_harness(torch.tensor([1], dtype=torch.float16), fw_op, True)
    type_out_harness(torch.tensor([1], dtype=torch.float32), fw_op, True)


#TODO(T29278) FIX
# The type will resolve to the same as input. In the float32, case, the ones
# will be wrongly generated as a float32, however the output type will remain
# the same
def test_rand_default_input_resolved_always_float16():
    def fw_op(input):
        return torch.rand(3, 5, 100, dtype=torch.float16) + input

    type_out_harness(torch.tensor([1], dtype=torch.float16), fw_op, True)
    type_out_harness(torch.tensor([1], dtype=torch.float32), fw_op, False)


## torch.normal tests ##


# The type will be resolved correctly as the mean and standard deviation are
# inputs to the op
def test_normal_mean_correctly_resolved():
    def fw_op(input_mean):
        return torch.normal(input_mean, 10.0)

    type_out_harness(torch.tensor([0.0], dtype=torch.float16), fw_op, True)
    type_out_harness(torch.tensor([0.0], dtype=torch.float32), fw_op, True)


# The type will be resolved correctly as the mean and standard deviation are
# inputs to the op
def test_normal_std_correctly_resolved():
    def fw_op(input_std):
        return torch.normal(0.0, input_std)

    type_out_harness(torch.tensor([10.0], dtype=torch.float16), fw_op, True)
    type_out_harness(torch.tensor([10.0], dtype=torch.float32), fw_op, True)


## torch.distributions.uniform.Uniform tests ##


# The type will always resolve to float32 as it is traced to torch.rand without
# the low and high input tensors (which become dead code)
def test_distributions_uniform():
    def fw_op(input_low):
        torch.manual_seed(42)
        ud = torch.distributions.uniform.Uniform(
            input_low, torch.tensor([10.0], dtype=torch.float32))
        return ud.sample((10, 10, 1000))

    type_out_harness(torch.tensor([1], dtype=torch.float16), fw_op, False)
    type_out_harness(torch.tensor([1], dtype=torch.float32), fw_op, True)


## torch.distributions.Normal tests ##


# The type will resolve correctly because the mean is an input
def test_distributions_normal_mean_correctly_resolved():
    def fw_op(input_mean):
        torch.manual_seed(42)
        ud = torch.distributions.Normal(input_mean, 10.0)
        return ud.sample((10, 10, 100))

    type_out_harness(torch.tensor([0.0], dtype=torch.float16), fw_op, True)
    type_out_harness(torch.tensor([0.0], dtype=torch.float32), fw_op, True)


def test_distributions_normal_std_correctly_resolved():
    def fw_op(input_std):
        torch.manual_seed(42)
        ud = torch.distributions.Normal(0.0, input_std)
        return ud.sample((10, 10, 100))

    type_out_harness(torch.tensor([10.0], dtype=torch.float16), fw_op, True)
    type_out_harness(torch.tensor([10.0], dtype=torch.float32), fw_op, True)


## tensor._uniform test #


# The type will resolve correctly because it is based on the empty tesnsor
def test_uniform_correctly_resolved():
    def fw_op(empty_tensor):
        torch.manual_seed(42)
        return empty_tensor.uniform_()

    type_out_harness(torch.empty((3, 4, 10), dtype=torch.float16), fw_op, True)
    type_out_harness(torch.empty((3, 4, 19), dtype=torch.float32), fw_op, True)


## tensor._normal test #


# The type will also resolve correctly because it is based on the empty tensor
def test_normal_correctly_resolved():
    def fw_op(empty_tensor):
        torch.manual_seed(42)
        return empty_tensor.normal_()

    type_out_harness(torch.empty((3, 4, 10), dtype=torch.float16), fw_op, True)
    type_out_harness(torch.empty((3, 4, 19), dtype=torch.float32), fw_op, True)


## tensor constant tests ##
#TODO(T29278) FIX
def test_constant_correctly_resolved():
    def fw_op(input):
        return torch.tensor([1, 2, 3], dtype=input.dtype) + input

    type_out_harness(torch.tensor([3, 4, 8], dtype=torch.float16), fw_op, True)
    type_out_harness(torch.tensor([3, 4, 8], dtype=torch.float32), fw_op, True)


def test_constant_add_float16():
    def fw_op(input):
        return torch.tensor([1, 2, 3], dtype=input.dtype) + input.to(
            torch.float16)

    type_out_harness(torch.tensor([3, 4, 8], dtype=torch.float16), fw_op, True)
    type_out_harness(torch.tensor([3, 4, 8], dtype=torch.float32), fw_op,
                     False)


def test_constant_always_float32():
    def fw_op(input):
        return torch.tensor([1, 2, 3], dtype=torch.float32) + input

    type_out_harness(torch.tensor([3, 4, 8], dtype=torch.float16), fw_op,
                     False)
    type_out_harness(torch.tensor([3, 4, 8], dtype=torch.float32), fw_op, True)


def test_float16_activations_float32_weights():
    torch.manual_seed(42)

    input = torch.ones(10)

    model = torch.nn.Linear(10, 20)

    # Float 32 act, float 32 weights
    pop_model = poptorch.inferenceModel(model, poptorch.Options())
    pop_out = pop_model(input)

    assert pop_out.dtype == torch.float

    # Float 16 act, float 32 weights
    pop_model = poptorch.inferenceModel(model, poptorch.Options())
    pop_out = pop_model(input.half())
    assert pop_out.dtype == torch.half


def test_master_weight_training():
    torch.manual_seed(42)

    model = torch.nn.Linear(10, 10)

    poptorch_model = helpers.trainingModelWithLoss(model,
                                                   loss=torch.nn.MSELoss())

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
    assert torch.allclose(out.float(), target, rtol=1e-02, atol=1e-02)


def test_bigger_model_training():
    torch.manual_seed(42)

    model = torch.nn.Sequential(torch.nn.Linear(10,
                                                10), torch.nn.Linear(10, 10),
                                torch.nn.Linear(10,
                                                10), torch.nn.Linear(10, 10),
                                torch.nn.Linear(10, 10))

    poptorch_model = helpers.trainingModelWithLoss(model,
                                                   loss=torch.nn.MSELoss())

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
    assert torch.allclose(out.float(), target, rtol=1e-02, atol=1e-02)
