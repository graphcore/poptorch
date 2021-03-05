#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

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


def execute_and_check_wrapper(model, input):
    # Run on CPU.
    native_out = model(input)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input)

    helpers.assert_allclose(actual=poptorch_out, expected=native_out)


@pytest.mark.parametrize("op", conv_1D)
@pytest.mark.parametrize("padding_mode", padding_modes)
def test_conv1D(op, padding_mode):
    if (op is torch.nn.ConvTranspose1d and padding_mode != 'zeros') or \
       padding_mode == 'circular': # TODO(T31811)
        pytest.skip('skipping unsupported padding_mode')
    torch.manual_seed(42)

    input = torch.randn(20, 4, 10)

    # With square kernels and equal stride
    model = op(4, 10, 3, stride=2, padding_mode=padding_mode)
    execute_and_check_wrapper(model, input)

    # Grouped convolutions.
    model = op(4, 8, 3, stride=2, groups=2, padding_mode=padding_mode)
    execute_and_check_wrapper(model, input)

    if op is not torch.nn.ConvTranspose1d:
        # # non-square kernels and unequal stride and with padding and dilation
        model = op(4,
                   33, (3),
                   stride=(2),
                   padding=(4),
                   dilation=(3),
                   padding_mode=padding_mode)
        execute_and_check_wrapper(model, input)


@pytest.mark.parametrize("op", conv_2D)
@pytest.mark.parametrize("padding_mode", padding_modes)
def test_conv2D(op, padding_mode):
    if (op is torch.nn.ConvTranspose2d and padding_mode != 'zeros') or \
       padding_mode == 'circular': # TODO(T31811)
        pytest.skip('skipping unsupported padding_mode')
    torch.manual_seed(42)

    input = torch.randn(20, 16, 50, 10)

    # With square kernels and equal stride
    model = op(16, 4, 3, stride=2, padding_mode=padding_mode)
    execute_and_check_wrapper(model, input)

    # Grouped convolutions.
    model = op(16, 4, (3, 5), stride=2, groups=2, padding_mode=padding_mode)
    execute_and_check_wrapper(model, input)

    # Rectangular padding/stride
    if op is not torch.nn.ConvTranspose2d:
        # non-square kernels and unequal stride and with padding
        model = op(16, 4, (3, 5), stride=(2, 1), padding=(4, 2))
        execute_and_check_wrapper(model, input)

        # non-square kernels and unequal stride and with padding and dilation
        model = op(16,
                   4, (3, 5),
                   stride=(2, 1),
                   padding=(4, 2),
                   dilation=(3),
                   padding_mode=padding_mode)
        execute_and_check_wrapper(model, input)


@pytest.mark.parametrize("op", conv_3D)
@pytest.mark.parametrize("padding_mode", padding_modes)
def test_conv3D(op, padding_mode):
    if (op is torch.nn.ConvTranspose3d and padding_mode != 'zeros') or \
       (op is torch.nn.Conv3d and padding_mode == 'reflect') or \
       padding_mode == 'circular': # TODO(T31811)
        pytest.skip('skipping unsupported padding_mode')

    torch.manual_seed(42)
    input = torch.randn(2, 4, 3, 5, 10)

    # With square kernels and equal stride
    model = op(4, 6, 3, stride=2, padding_mode=padding_mode)
    execute_and_check_wrapper(model, input)

    # Grouped convolutions.
    model = op(4, 6, 3, stride=2, groups=2, padding_mode=padding_mode)
    execute_and_check_wrapper(model, input)

    if op is not torch.nn.ConvTranspose3d:
        # non-square kernels and unequal stride and with padding
        model = op(4,
                   6, (3, 2, 2),
                   stride=(2, 1, 1),
                   padding=(4, 2, 0),
                   padding_mode=padding_mode)
        execute_and_check_wrapper(model, input)

        # non-square kernels and unequal stride and with padding and dilation
        model = op(4,
                   6, (3, 4, 2),
                   stride=(2, 1, 1),
                   padding=(4, 2, 0),
                   dilation=(3, 1, 1))

        execute_and_check_wrapper(model, input)


def test_available_memory():
    torch.manual_seed(42)
    input = torch.randn(2, 4, 3, 10)

    class BasicNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 4, 3, stride=2)

        def forward(self, x):
            out = self.conv(x)
            out = poptorch.set_available_memory(out, 0.6)
            return out

    # Just check we don't explode when the value is set.
    model = BasicNetwork()
    execute_and_check_wrapper(model, input)


@pytest.mark.parametrize("mode", poptorch.MatMulSerializationMode)
def test_matmul_serialization(mode):
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
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(lhs, rhs)

    helpers.assert_allclose(actual=poptorch_out, expected=native_out)


def test_available_memory_automatic():
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

    poptorch_model = poptorch.inferenceModel(model, opts)
    poptorch_out = poptorch_model(input)

    helpers.assert_allclose(actual=poptorch_out, expected=native_out)


@pytest.mark.parametrize("dim", range(-3, 3))
def test_cumsum(dim):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.cumsum(x, dim=dim)

    model = Model()
    torch.manual_seed(0)
    input = torch.randn(1, 5, 6, dtype=torch.float32)

    execute_and_check_wrapper(model, input)
