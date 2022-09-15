#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import helpers
import poptorch


def assert_same_type(inputs, model, opts, expect_same_type):
    native_out = model(inputs)

    pop_model = poptorch.inferenceModel(model, opts)
    pop_out = pop_model(inputs)

    if expect_same_type:
        assert native_out.dtype == pop_out.dtype
    else:
        assert native_out.dtype != pop_out.dtype


def type_out_harness(trace_model, inputs, forward_op,
                     expect_same_type_float_downcast_to_half,
                     expect_same_type_like_poptorch):
    class Model(torch.nn.Module):
        def forward(self, x):
            return forward_op(x)

    model = Model()
    opts = poptorch.Options()
    opts.Jit.traceModel(trace_model)

    if trace_model:
        opts.Precision.halfFloatCasting(
            poptorch.HalfFloatCastingBehavior.FloatDowncastToHalf)
        assert_same_type(inputs, model, opts,
                         expect_same_type_float_downcast_to_half)

    opts = opts.clone()
    if trace_model:
        opts.Precision.halfFloatCasting(
            poptorch.HalfFloatCastingBehavior.HalfUpcastToFloat)
    assert_same_type(inputs, model, opts, expect_same_type_like_poptorch)


## Ones and Zeros tests ##

ones_zeros = [torch.ones, torch.zeros]


# Tracing: input dtype always resolves to float32, because it is traced as float32,
# and the input itself is not used standalone.
#
# Dispatcher: The dtype will match what was requested.
@pytest.mark.parametrize("op", ones_zeros)
@pytest.mark.parametrize("trace_model", [True, False])
def test_ones_zeros_default_resolved(op, trace_model):
    def fw_op(input):
        return op((2, 3, 4), dtype=input.dtype,
                  device=helpers.outputDevice()) + input.to(input.dtype)

    if trace_model:
        float16_expected = False
    else:
        float16_expected = True

    type_out_harness(trace_model, torch.tensor([1], dtype=torch.float16),
                     fw_op, float16_expected, float16_expected)
    type_out_harness(trace_model, torch.tensor([1], dtype=torch.float32),
                     fw_op, True, True)


# The dtype will correctly resolve becuse it matches the input added
# All settings will match pytorch
@pytest.mark.parametrize("op", ones_zeros)
@pytest.mark.parametrize("trace_model", [True, False])
def test_ones_zeros_input_resolved_with_input_dtype(op, trace_model):
    def fw_op(input):
        return op((2, 3, 4), dtype=input.dtype,
                  device=helpers.outputDevice()) + input

    type_out_harness(trace_model, torch.tensor([1], dtype=torch.float16),
                     fw_op, True, True)
    type_out_harness(trace_model, torch.tensor([1], dtype=torch.float32),
                     fw_op, True, True)


# The zeros/ones will resolve correctly becaue torch.float16 could not have been
# from a tensor which could have beeh half/float.
#
# Half and half to float:
# The output will always be float 16.
#
# Like pytorch:
# The output will be correct.


@pytest.mark.parametrize("op", ones_zeros)
@pytest.mark.parametrize("trace_model", [True, False])
def test_ones_zeros_input_resolved_always_float16(op, trace_model):
    def fw_op(input):
        return op(
            (2, 3, 4), dtype=torch.float16,
            device=helpers.outputDevice()) + input

    type_out_harness(trace_model, torch.tensor([1], dtype=torch.float16),
                     fw_op, True, True)
    type_out_harness(trace_model, torch.tensor([1], dtype=torch.float32),
                     fw_op, False, True)


# The dtype will resolve to the same as input. In the float16 case, the
# ones/zeros will be wrongly generated as a float16.
#
# The output will always match input.
@pytest.mark.parametrize("op", ones_zeros)
@pytest.mark.parametrize("trace_model", [True, False])
def test_ones_zeros_input_resolved_always_float32(op, trace_model):
    def fw_op(input):
        return op(
            (2, 3, 4), dtype=torch.float32,
            device=helpers.outputDevice()) + input

    type_out_harness(trace_model, torch.tensor([1], dtype=torch.float16),
                     fw_op, False, not trace_model)
    type_out_harness(trace_model, torch.tensor([1], dtype=torch.float32),
                     fw_op, True, True)


## torch.rand tests ##


# Tracing: The dtype will always resolve to float32 as tracing always happens with
# float 32 and the input is not captured.
#
# Dispatcher: The dtype will match what was requested.
@pytest.mark.parametrize("trace_model", [True, False])
def test_rand_default_resolved(trace_model):
    def fw_op(input):
        return torch.rand(3, 5, 100, dtype=input.dtype)

    if trace_model:
        float16_expected = False
    else:
        float16_expected = True

    type_out_harness(trace_model, torch.tensor([1], dtype=torch.float16),
                     fw_op, float16_expected, float16_expected)
    type_out_harness(trace_model, torch.tensor([1], dtype=torch.float32),
                     fw_op, True, True)


#The dtype will correctly resolve becuse it matches the input added
@pytest.mark.parametrize("trace_model", [True, False])
def test_rand_default_input_resolved(trace_model):
    def fw_op(input):
        return torch.rand(3, 5, 100, dtype=input.dtype) + input

    type_out_harness(trace_model, torch.tensor([1], dtype=torch.float16),
                     fw_op, True, True)
    type_out_harness(trace_model, torch.tensor([1], dtype=torch.float32),
                     fw_op, True, True)


# The type will resolve correctly because torch.float16 could not have been
# from a tensor which could have been half/float.
#
# Half and half to float:
# The output will always be float 16.
#
# Like pytorch:
# The output will be correct.
@pytest.mark.parametrize("trace_model", [True, False])
def test_rand_default_input_resolved_always_float16(trace_model):
    def fw_op(input):
        return torch.rand(3, 5, 100, dtype=torch.float16) + input

    type_out_harness(trace_model, torch.tensor([1], dtype=torch.float16),
                     fw_op, True, True)
    type_out_harness(trace_model, torch.tensor([1], dtype=torch.float32),
                     fw_op, False, True)


## torch.normal tests ##


# The type will be resolved correctly as the mean and standard deviation are
# inputs to the op
@pytest.mark.parametrize("trace_model", [True, False])
def test_normal_mean_correctly_resolved(trace_model):
    def fw_op(input_mean):
        return torch.normal(input_mean, 10.0)

    type_out_harness(trace_model, torch.tensor([0.0], dtype=torch.float16),
                     fw_op, True, True)
    type_out_harness(trace_model, torch.tensor([0.0], dtype=torch.float32),
                     fw_op, True, True)


# The type will be resolved correctly as the mean and standard deviation are
# inputs to the op
@pytest.mark.parametrize("trace_model", [True, False])
def test_normal_std_correctly_resolved(trace_model):
    def fw_op(input_std):
        return torch.normal(0.0, input_std)

    type_out_harness(trace_model, torch.tensor([10.0], dtype=torch.float16),
                     fw_op, True, True)
    type_out_harness(trace_model, torch.tensor([10.0], dtype=torch.float32),
                     fw_op, True, True)


## torch.distributions.uniform.Uniform tests ##


# The type will always resolve to float32 as it is traced to torch.rand without
# the low and high input tensors (which become dead code)
@pytest.mark.parametrize("trace_model", [True, False])
def test_distributions_uniform(trace_model):
    def fw_op(input_low):
        torch.manual_seed(42)
        ud = torch.distributions.uniform.Uniform(
            input_low, torch.tensor([10.0], dtype=torch.float32))
        return ud.sample((10, 10, 1000))

    type_out_harness(trace_model, torch.tensor([1], dtype=torch.float16),
                     fw_op, False, not trace_model)
    type_out_harness(trace_model, torch.tensor([1], dtype=torch.float32),
                     fw_op, True, True)


## torch.distributions.Normal tests ##


# The type will resolve correctly because the mean is an input
@pytest.mark.parametrize("trace_model", [True, False])
def test_distributions_normal_mean_correctly_resolved(trace_model):
    def fw_op(input_mean):
        torch.manual_seed(42)
        ud = torch.distributions.Normal(input_mean, 10.0)
        return ud.sample((10, 10, 100))

    type_out_harness(trace_model, torch.tensor([0.0], dtype=torch.float16),
                     fw_op, True, True)
    type_out_harness(trace_model, torch.tensor([0.0], dtype=torch.float32),
                     fw_op, True, True)


@pytest.mark.parametrize("trace_model", [True, False])
def test_distributions_normal_std_correctly_resolved(trace_model):
    def fw_op(input_std):
        torch.manual_seed(42)
        ud = torch.distributions.Normal(0.0, input_std)
        return ud.sample((10, 10, 100))

    type_out_harness(trace_model, torch.tensor([10.0], dtype=torch.float16),
                     fw_op, True, True)
    type_out_harness(trace_model, torch.tensor([10.0], dtype=torch.float32),
                     fw_op, True, True)


## tensor._uniform test #


# The type will resolve correctly because it is based on the input tensor
@pytest.mark.parametrize("trace_model", [True, False])
def test_uniform_correctly_resolved(trace_model):
    def fw_op(input_tensor):
        torch.manual_seed(42)
        input_tensor = input_tensor + 0  # Ensure input is not modified in place
        return input_tensor.uniform_()

    type_out_harness(trace_model, torch.empty((3, 4, 10), dtype=torch.float16),
                     fw_op, True, True)
    type_out_harness(trace_model, torch.empty((3, 4, 10), dtype=torch.float32),
                     fw_op, True, True)


## tensor._normal test #


# The type will also resolve correctly because it is based on the input tensor
@pytest.mark.parametrize("trace_model", [True, False])
def test_normal_correctly_resolved(trace_model):
    def fw_op(input_tensor):
        torch.manual_seed(42)
        input_tensor = input_tensor + 0  # Ensure input is not modified in place
        return input_tensor.normal_()

    type_out_harness(trace_model, torch.empty((3, 4, 10), dtype=torch.float16),
                     fw_op, True, True)
    type_out_harness(trace_model, torch.empty((3, 4, 10), dtype=torch.float32),
                     fw_op, True, True)


## tensor constant tests ##


# The type will resolve correctly because it is added to the input.
#
# The output will always be the same as the
@pytest.mark.parametrize("trace_model", [True, False])
def test_constant_correctly_resolved(trace_model):
    def fw_op(input):
        return torch.tensor([1, 2, 3], dtype=input.dtype) + input

    type_out_harness(trace_model, torch.tensor([3, 4, 8], dtype=torch.float16),
                     fw_op, True, True)
    type_out_harness(trace_model, torch.tensor([3, 4, 8], dtype=torch.float32),
                     fw_op, True, True)


# The type will resolve to float16 always because the input is cast to float16
# The output will always be float 16.
@pytest.mark.parametrize("trace_model", [True, False])
def test_constant_add_float16(trace_model):
    def fw_op(input):
        return torch.tensor([1, 2, 3], dtype=input.dtype) + input.to(
            torch.float16)

    type_out_harness(trace_model, torch.tensor([3, 4, 8], dtype=torch.float16),
                     fw_op, True, True)
    type_out_harness(trace_model, torch.tensor([3, 4, 8], dtype=torch.float32),
                     fw_op, False, not trace_model)


# The type will resolve to the input rather than float32 because of the
# ambiguity betwen tracing with a float and a half converted to a float.
@pytest.mark.parametrize("trace_model", [True, False])
def test_constant_always_float32(trace_model):
    def fw_op(input):
        return torch.tensor([1, 2, 3], dtype=torch.float32) + input

    type_out_harness(trace_model, torch.tensor([3, 4, 8], dtype=torch.float16),
                     fw_op, False, not trace_model)
    type_out_harness(trace_model, torch.tensor([3, 4, 8], dtype=torch.float32),
                     fw_op, True, True)


@pytest.mark.parametrize("trace_model", [True, False])
@pytest.mark.parametrize("conv", [True, False])
def test_float16_activations_float32_weights(trace_model, conv):
    torch.manual_seed(42)

    if conv:
        input = torch.ones(1, 4, 4)
        model = torch.nn.Conv1d(4, 5, 2)
    else:
        input = torch.ones(10)
        model = torch.nn.Linear(10, 20)

    # Float 32 act, float 32 weights
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    pop_model = poptorch.inferenceModel(model, options)
    pop_out = pop_model(input)

    assert pop_out.dtype == torch.float

    # Float 16 act, float 32 weights
    pop_model = poptorch.inferenceModel(model, options)
    pop_out = pop_model(input.half())
    assert pop_out.dtype == torch.half

    # Float 32 act, float 16 weights
    model.half()
    pop_model = poptorch.inferenceModel(model, options)
    pop_out = pop_model(input)
    if trace_model and not conv:
        assert pop_out.dtype == torch.half
    else:
        assert pop_out.dtype == torch.float

    # Float 16 act, float 16 weights
    pop_model = poptorch.inferenceModel(model, options)
    pop_out = pop_model(input.half())
    assert pop_out.dtype == torch.half


@pytest.mark.parametrize("trace_model", [True, False])
def test_master_weight_training(trace_model):
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
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=options)

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


@pytest.mark.parametrize("trace_model", [True, False])
def test_bigger_model_training(trace_model):
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
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=options)

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
