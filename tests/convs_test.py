#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import unittest.mock
import torch
import pytest
import poptorch
import helpers

# Convolutions.

convolutions = [
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
    torch.nn.Unfold,
    torch.nn.Fold,
]

padding_modes = ['zeros', 'reflect', 'replicate', 'circular']

# Unsupported
folds = []  # torch.nn.Unfold, torch.nn.Fold,

# Supported.
conv_1D = [torch.nn.Conv1d, torch.nn.ConvTranspose1d]
conv_2D = [torch.nn.Conv2d, torch.nn.ConvTranspose2d]
conv_3D = [torch.nn.Conv3d, torch.nn.ConvTranspose3d]


def execute_and_check_wrapper(trace_model,
                              op,
                              input,
                              training=True,
                              rtol=0.01,
                              atol=0.01):
    # TODO(T25617): PopART does not support PadGradOp when mode is not
    # "constant"
    if hasattr(op, 'padding_mode') and op.padding_mode != 'zeros':
        return

    model = helpers.ModelWithWeights(op,
                                     input.shape,
                                     loss_fn=torch.nn.L1Loss(reduction='mean'),
                                     out_fn=lambda x: (x, torch.zeros_like(x)))

    if training:
        optimizer = poptorch.optim.SGD(model.parameters(), lr=0.01)
        poptorch_model = poptorch.trainingModel(model, optimizer=optimizer)

        try:
            has_own_weight = any([
                n == 'weight'
                for (n, p) in poptorch_model.op.named_parameters()
            ])
        except AttributeError:
            has_own_weight = False

        if has_own_weight:
            weights_before = poptorch_model.op.weight.detach().clone()

        input = torch.ones_like(input)
        for _ in range(5):
            poptorch_out, loss = poptorch_model((input, ))

        if has_own_weight:
            model.op.weight.data = weights_before

        # pylint: disable=protected-access
        model.lin.weight.data = model._weights_before
        for _ in range(5):
            optimizer.zero_grad()
            native_out, loss = model((input, ))
            loss.backward()
            optimizer.step()

        # Inference test - check outputs
        helpers.assert_allclose(actual=poptorch_out,
                                expected=native_out,
                                rtol=rtol,
                                atol=atol)
    else:
        options = poptorch.Options()
        options.Jit.traceModel(trace_model)
        poptorch_model = poptorch.inferenceModel(model, options)
        # Run on CPU.
        native_out, _ = model((input, ))

        # Run on IPU.
        poptorch_out, _ = poptorch_model((input, ))
        helpers.assert_allclose(actual=poptorch_out,
                                expected=native_out,
                                rtol=rtol,
                                atol=atol)


@pytest.mark.parametrize("op", conv_1D)
@pytest.mark.parametrize("padding_mode", padding_modes)
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_conv1D(op, padding_mode, training, trace_model):
    if (op is torch.nn.ConvTranspose1d and padding_mode != 'zeros') or \
       padding_mode == 'circular': # TODO(T31811)
        pytest.skip('skipping unsupported padding_mode')
    torch.manual_seed(42)
    C_IN = 4
    C_OUT = 8

    input = torch.randn(1, C_IN, 10)
    # With square kernels and equal stride
    model = op(C_IN, C_OUT, 3, stride=2, padding_mode=padding_mode)
    execute_and_check_wrapper(trace_model, model, input, training)

    if op is not torch.nn.ConvTranspose1d:
        # non-square kernels and unequal stride and with padding and dilation
        model = op(C_IN,
                   C_OUT, (3),
                   stride=(2),
                   padding=(4),
                   dilation=(3),
                   padding_mode=padding_mode)
        execute_and_check_wrapper(trace_model, model, input, training)


@pytest.mark.parametrize("op", conv_2D)
@pytest.mark.parametrize("padding_mode", padding_modes)
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_conv2D(op, padding_mode, training, trace_model):
    if (op is torch.nn.ConvTranspose2d and padding_mode != 'zeros') or \
       padding_mode == 'circular': # TODO(T31811)
        pytest.skip('skipping unsupported padding_mode')
    torch.manual_seed(42)
    C_IN = 4
    C_OUT = 2
    input = torch.randn(1, C_IN, 8, 10)

    # With square kernels and equal stride
    model = op(C_IN, C_OUT, 3, stride=2, padding_mode=padding_mode)
    execute_and_check_wrapper(trace_model,
                              model,
                              input,
                              training,
                              rtol=0.1,
                              atol=0.1)

    # Grouped convolutions.

    model = op(C_IN,
               C_OUT, (3, 5),
               stride=2,
               groups=2,
               padding_mode=padding_mode)
    execute_and_check_wrapper(trace_model,
                              model,
                              input,
                              training,
                              rtol=0.1,
                              atol=0.1)

    # Rectangular padding/stride
    if op is not torch.nn.ConvTranspose2d:
        # non-square kernels and unequal stride and with padding
        model = op(C_IN, C_OUT, (3, 5), stride=(2, 1), padding=(4, 2))
        execute_and_check_wrapper(trace_model, model, input, training=False)

        # non-square kernels and unequal stride and with padding and dilation
        model = op(C_IN,
                   C_OUT, (3, 5),
                   stride=(2, 1),
                   padding=(4, 2),
                   dilation=(3),
                   padding_mode=padding_mode)
        execute_and_check_wrapper(trace_model,
                                  model,
                                  input,
                                  training,
                                  rtol=0.01,
                                  atol=0.05)


@pytest.mark.parametrize("op", conv_3D)
@pytest.mark.parametrize("padding_mode", padding_modes)
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_conv3D(op, padding_mode, training, trace_model):
    if (op is torch.nn.ConvTranspose3d and padding_mode != 'zeros') or \
       (op is torch.nn.Conv3d and padding_mode == 'reflect') or \
       padding_mode == 'circular': # TODO(T31811)
        pytest.skip('skipping unsupported padding_mode')

    torch.manual_seed(42)
    C_IN = 4
    C_OUT = 2
    input = torch.randn(1, C_IN, 3, 5, 8)

    # With square kernels and equal stride
    model = op(C_IN, C_OUT, 3, stride=2, padding_mode=padding_mode)
    execute_and_check_wrapper(trace_model,
                              model,
                              input,
                              training,
                              rtol=0.1,
                              atol=0.1)

    # Grouped convolutions.
    model = op(C_IN, C_OUT, 3, stride=2, groups=2, padding_mode=padding_mode)
    execute_and_check_wrapper(trace_model,
                              model,
                              input,
                              training,
                              rtol=0.1,
                              atol=0.1)

    if op is torch.nn.ConvTranspose3d:
        #  test output padding
        model = op(C_IN,
                   C_OUT, (3, 2, 2),
                   stride=(2, 1, 1),
                   groups=2,
                   output_padding=[1, 0, 0],
                   padding_mode=padding_mode)
        execute_and_check_wrapper(trace_model,
                                  model,
                                  input,
                                  training,
                                  rtol=0.05,
                                  atol=0.05)
    else:
        # non-square kernels and unequal stride and with padding
        model = op(C_IN,
                   C_OUT, (3, 2, 2),
                   stride=(2, 1, 1),
                   padding=(4, 2, 0),
                   padding_mode=padding_mode)

        execute_and_check_wrapper(trace_model,
                                  model,
                                  input,
                                  training,
                                  rtol=0.1,
                                  atol=0.1)

        # non-square kernels and unequal stride and with padding and dilation
        model = op(C_IN,
                   C_OUT, (3, 4, 2),
                   stride=(2, 1, 1),
                   padding=(4, 2, 0),
                   dilation=(3, 1, 1))

        execute_and_check_wrapper(trace_model,
                                  model,
                                  input,
                                  training,
                                  rtol=0.1,
                                  atol=0.1)


# The test is reliant on an IPU model with limited memory, so force the small model
@unittest.mock.patch.dict("os.environ", helpers.forceSmallModel())
@pytest.mark.parametrize("trace_model", [True, False])
def test_available_memory(trace_model):
    torch.manual_seed(42)
    input = torch.randn(1, 4, 10, 10)

    model = torch.nn.Conv2d(4, 1576, 10, stride=1)

    # Test that the small IPU model runs out of memory without AMP
    with pytest.raises(poptorch.Error,
                       match="receives more data than it has total memory"):
        execute_and_check_wrapper(trace_model, model, input)

    model.register_forward_hook(lambda _1, _2, conv: poptorch.
                                set_available_memory(conv, 0.5))
    # Test that AMP fixes the OOM error
    execute_and_check_wrapper(trace_model, model, input)


@pytest.mark.parametrize("mode", poptorch.MatMulSerializationMode)
@pytest.mark.parametrize("trace_model", [True, False])
def test_matmul_serialization(mode, trace_model):
    torch.manual_seed(42)

    input_channels = 6
    reducing_dim = 2
    output_channels = 4
    lhs = torch.randn(input_channels, reducing_dim)
    rhs = torch.randn(reducing_dim, output_channels)
    if mode == poptorch.MatMulSerializationMode.Disabled:
        factor = 0
    elif mode == poptorch.MatMulSerializationMode.InputChannels:
        factor = 2
    elif mode == poptorch.MatMulSerializationMode.ReducingDim:
        factor = 2
    elif mode == poptorch.MatMulSerializationMode.OutputChannels:
        factor = 4
    else:
        assert False, "Invalid mode"

    class BasicNetwork(torch.nn.Module):
        def forward(self, x, y):
            out = poptorch.serializedMatMul(x,
                                            y,
                                            mode,
                                            factor,
                                            keep_precision=True)
            return out

    # Just check we don't explode when the value is set.
    model = BasicNetwork()
    native_out = model(lhs, rhs)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    poptorch_out = poptorch_model(lhs, rhs)

    helpers.assert_allclose(actual=poptorch_out, expected=native_out)


@pytest.mark.parametrize("trace_model", [True, False])
def test_available_memory_automatic(trace_model):
    torch.manual_seed(42)

    # Just check we don't explode when the value is set.
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 10, 5),
                                              torch.nn.MaxPool2d(2),
                                              torch.nn.ReLU())
            self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(10, 20, 5),
                                              torch.nn.MaxPool2d(2),
                                              torch.nn.ReLU())
            self.layer3 = torch.nn.Linear(320, 256)
            self.layer3_act = torch.nn.ReLU()
            self.layer4 = torch.nn.Linear(256, 10)

            self.softmax = torch.nn.LogSoftmax(1)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = x.view(-1, 320)

            x = self.layer3_act(self.layer3(x))
            x = self.layer4(x)
            x = self.softmax(x)
            return x

    model = Network()
    # Run on CPU.
    input = torch.randn(2, 1, 28, 28)
    native_out = model(input)

    # Run on IPU.
    opts = poptorch.Options()
    opts.setAvailableMemoryProportion(available_memory_proportion={
        "IPU0": 0.7,
        "IPU1": 0.2
    })
    opts.Jit.traceModel(trace_model)

    poptorch_model = poptorch.inferenceModel(model, opts)
    poptorch_out = poptorch_model(input)

    helpers.assert_allclose(actual=poptorch_out, expected=native_out)


@pytest.mark.parametrize("dim", range(-3, 3))
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_cumsum(dim, training, trace_model):
    torch.manual_seed(42)

    op = lambda x: torch.cumsum(x, dim=dim)
    input = torch.randn(1, 5, 6, dtype=torch.float32)

    execute_and_check_wrapper(trace_model,
                              op,
                              input,
                              training,
                              rtol=0.02,
                              atol=0.02)


@pytest.mark.parametrize("src_dtype", [torch.float, torch.int])
@pytest.mark.parametrize("dest_dtype", [torch.float, torch.int])
@pytest.mark.parametrize("dim", range(-1, 1))
def test_cumsum_changing_types(src_dtype, dest_dtype, dim):
    class Model(torch.nn.Module):
        def forward(self, inp):
            return inp.cumsum(dim=dim, dtype=dest_dtype)

    cpu_model = Model()
    ipu_model = poptorch.inferenceModel(cpu_model)

    torch.manual_seed(42)
    inp = torch.randn(1, 5, 6).to(src_dtype)

    helpers.assert_allclose(actual=ipu_model(inp), expected=cpu_model(inp))


# The free-function, `out=` form of `cumsum` works a bit differently to the
# method form.
@pytest.mark.parametrize("src_dtype", [torch.float, torch.int])
@pytest.mark.parametrize("dest_dtype", [torch.float, torch.int])
@pytest.mark.parametrize("dim", range(-1, 1))
def test_cumsum_changing_types_out(src_dtype, dest_dtype, dim):
    class Model(torch.nn.Module):
        def forward(self, inp):
            res = torch.empty(inp.shape).to(dest_dtype)
            return torch.cumsum(inp, dim=dim, out=res)

    cpu_model = Model()
    ipu_model = poptorch.inferenceModel(cpu_model)

    torch.manual_seed(42)
    inp = torch.randn(1, 5, 6).to(src_dtype)

    helpers.assert_allclose(actual=ipu_model(inp), expected=cpu_model(inp))


# Test that the result of `cumsum` can be passed forward without loss of tensor
# shape metadata.
@pytest.mark.parametrize("src_dtype", [torch.float, torch.int])
@pytest.mark.parametrize("dest_dtype", [torch.float, torch.int])
@pytest.mark.parametrize("dim", range(-1, 1))
def test_cumsum_can_pass_on(src_dtype, dest_dtype, dim):
    class Model(torch.nn.Module):
        def forward(self, inp):
            return inp.cumsum(dim=dim, dtype=dest_dtype)[:, -1]

    ipu_model = poptorch.inferenceModel(Model())

    torch.manual_seed(42)
    inp = torch.randn(1, 5, 6).to(src_dtype)

    # Just test it doesn't fail
    try:
        ipu_model(inp)
    except poptorch.poptorch_core.Error as _:
        assert False, "Passing the result of torch.cumsum onwards failed."
