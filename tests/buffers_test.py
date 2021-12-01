#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import pytest

import poptorch
import helpers


class ConstantBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('stuff', torch.tensor([1, 2, 3],
                                                   dtype=torch.int32))

    def forward(self, x):
        new_stuff = 1.0 + self.stuff
        return torch.sum(x + new_stuff)


@pytest.mark.parametrize("trace_model", [True, False])
def test_constant_buffer(trace_model):
    model = ConstantBuffer()

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    assert poptorch_model(torch.tensor([2])) == 15


@pytest.mark.parametrize("trace_model", [True, False])
def test_constant_buffer_repeat(trace_model):
    model = ConstantBuffer()

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    assert poptorch_model(torch.tensor([2])) == 15
    assert poptorch_model(torch.tensor([2])) == 15


def test_buffer_implicit_copy():
    momentum = 0.1

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.bn = torch.nn.BatchNorm1d(10, momentum=momentum)
            self.loss = torch.nn.MSELoss()

        def forward(self, x, target):
            y = self.bn(x)
            return y, self.loss(y, target)

    model = Model()

    input = torch.ones([4, 10], dtype=torch.float32)
    target = torch.ones([4, 10], dtype=torch.float32) + 1

    poptorch_model = poptorch.trainingModel(model)

    poptorch_model(input, target)
    helpers.assert_allclose(actual=model.bn.running_mean,
                            expected=input[0, :] * momentum)

    poptorch_model.copyWeightsToHost()
    helpers.assert_allclose(actual=model.bn.running_mean,
                            expected=input[0, :] * momentum)


@pytest.mark.parametrize("trace_model", [True, False])
def test_error_on_remove_buffer(trace_model):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('y', torch.tensor([2]))

        def forward(self, x):
            x = x + 1
            if 'y' in self._buffers:
                del self._buffers['y']
            return x

    model = Model()

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)

    error_msg = (r"Buffer y is removed from the model when calling the " +
                 r"forward method\.")
    with pytest.raises(poptorch.Error, match=error_msg):
        poptorch_model(torch.tensor([5.0]))


@pytest.mark.parametrize("trace_model", [True, False])
def test_error_on_redefine_buffer(trace_model):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('y', torch.tensor([2]))

        def forward(self, x):
            x = x + 1
            # pylint: disable=attribute-defined-outside-init
            self.y = x

    model = Model()

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    error_msg = (r"Buffer y is reassigned within the model when calling the " +
                 r"forward method\. This is not supported\. Consider using " +
                 r"self\.y\.copy_\(src\) to copy data " +
                 r"from a source tensor, where src is the name of the " +
                 r"source tensor\.")

    with pytest.raises(poptorch.Error, match=error_msg):
        poptorch_model(torch.tensor([5.0]))


class BufferUpdatingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 1, padding=0)
        self.register_buffer("test_buff", torch.zeros([2],
                                                      dtype=torch.float32))

        self.loss = torch.nn.L1Loss()

    def forward(self, inp, target):
        x = self.conv(inp)

        with torch.no_grad():
            # pylint: disable=attribute-defined-outside-init
            self.test_buff += self.conv.bias[0]

        return x, self.loss(x, target)


@pytest.mark.parametrize("device_iterations", [1, 3, 5])
@pytest.mark.parametrize("gradient_accumulation", [1, 3, 5])
def test_buffer_update_with_param(device_iterations, gradient_accumulation):
    model = BufferUpdatingModel()
    model.conv.weight.data = torch.ones_like(model.conv.weight.data)
    model.conv.bias.data = torch.ones_like(model.conv.bias.data)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    times_to_run = 10
    dummy_input = torch.ones([2, 2, 2, 2])
    dummy_target = torch.zeros_like(dummy_input)

    for _ in range(times_to_run * device_iterations):
        opt.zero_grad()
        for _ in range(gradient_accumulation):
            _, loss = model(dummy_input, dummy_target)

            # Match mean gradient_accumulation
            loss /= gradient_accumulation

            loss.backward()

        opt.step()

    model_bias = model.conv.bias.clone()
    model_test_buff = model.test_buff.clone()

    # pylint: disable=attribute-defined-outside-init
    model.test_buff = torch.zeros([2], dtype=torch.float32)
    model.conv.weight.data = torch.ones_like(model.conv.weight.data)
    model.conv.bias.data = torch.ones_like(model.conv.bias.data)

    # Check for proper cloning
    with pytest.raises(AssertionError):
        helpers.assert_allclose(expected=model_bias, actual=model.conv.bias)
    with pytest.raises(AssertionError):
        helpers.assert_allclose(expected=model_test_buff,
                                actual=model.test_buff)

    opts = poptorch.Options()
    opts.deviceIterations(device_iterations)
    opts.Training.gradientAccumulation(gradient_accumulation)

    dummy_input = torch.ones(
        [2 * device_iterations * gradient_accumulation, 2, 2, 2])
    dummy_target = torch.zeros_like(dummy_input)

    poptorch_model = poptorch.trainingModel(model,
                                            optimizer=torch.optim.SGD(
                                                model.parameters(), lr=0.1),
                                            options=opts)

    for _ in range(times_to_run):
        dummy_target = torch.zeros_like(dummy_input)
        poptorch_model(dummy_input, dummy_target)

    helpers.assert_allclose(expected=model_bias,
                            actual=poptorch_model.conv.bias)
    helpers.assert_allclose(expected=model_test_buff,
                            actual=poptorch_model.test_buff)


def test_failing_on_replicas():
    model = BufferUpdatingModel()

    opts = poptorch.Options()
    opts.replicationFactor(2)
    poptorch_model = poptorch.trainingModel(model,
                                            optimizer=torch.optim.SGD(
                                                model.parameters(), lr=0.1),
                                            options=opts)

    dummy_input = torch.ones([4, 2, 2, 2])
    dummy_target = torch.zeros_like(dummy_input)

    error_msg = (r"PopTorch does not support broadcasting buffers. " +
                 r"If your model is able to tolerate buffers becoming " +
                 r"out of sync between replicas, you can disable " +
                 r"buffer broadcasting using " +
                 r"poptorch.Options.broadcastBuffers\(False\).")

    with pytest.raises(poptorch.Error, match=error_msg):
        poptorch_model(dummy_input, dummy_target)


@pytest.mark.parametrize("trace_model", [True, False])
def test_constant_buffer_with_replicas(trace_model):
    # This should not have an error as the buffer is constant
    model = ConstantBuffer()

    opts = poptorch.Options()
    opts.replicationFactor(2)
    opts.Jit.traceModel(trace_model)

    poptorch_model = poptorch.inferenceModel(model, opts)
    poptorch_model(torch.tensor([1, 2]))


@pytest.mark.parametrize("trace_model", [True, False])
def test_no_input_but_one_buffer(trace_model):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("x", torch.tensor([1.], dtype=torch.float))

        def forward(self):
            # pylint: disable=attribute-defined-outside-init,no-member
            self.x += 1.0
            return self.x

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)

    assert poptorch_model() == 2.
    assert poptorch_model() == 3.
    assert poptorch_model() == 4.
    assert poptorch_model() == 5.


@pytest.mark.parametrize("trace_model", [True, False])
def test_unsynchronised_replicated_buffers(trace_model):
    class ReplicaBufferModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.zeros(1, 2))

        def forward(self, x):
            buffer_update = self.buffer + x
            self.buffer.copy_(buffer_update)
            return poptorch.identity_loss(self.buffer, reduction='none')

    num_replica = 2
    torch.manual_seed(43)
    opts = poptorch.Options()
    opts.replicationFactor(num_replica)
    opts.deviceIterations(1)
    opts.broadcastBuffers(False)
    opts.Jit.traceModel(trace_model)

    model = ReplicaBufferModel()
    model.float()
    poptorch_model = poptorch.inferenceModel(model, opts)

    x = torch.tensor([[9], [2]])

    # Each replica update its buffer in place with a random value 50 times.
    for _ in range(50):
        y = poptorch_model(x)

    assert y[0][-1] == x[0] * 50
    assert y[1][-1] == x[1] * 50
