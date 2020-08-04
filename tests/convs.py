#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch

import pytest

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

# TODO(T22980):
# padding_mode (string, optional) - 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros

# Unsupported
folds = []  # torch.nn.Unfold, torch.nn.Fold,

# Supported.
conv_1D = [torch.nn.Conv1d]  # torch.nn.ConvTranspose1d]
conv_2D = [torch.nn.Conv2d]  # torch.nn.ConvTranspose2d]
conv_3D = [torch.nn.Conv3d]  # torch.nn.ConvTranspose3d]


def execute_and_check_wrapper(model, input):
    # Run on CPU.
    nativeOut = model(input)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input)

    torch.testing.assert_allclose(poptorch_out, nativeOut)


@pytest.mark.parametrize("op", conv_1D)
def test_conv1D(op):

    torch.manual_seed(42)

    input = torch.randn(20, 16, 10)

    # With square kernels and equal stride
    model = op(16, 33, 3, stride=2)
    execute_and_check_wrapper(model, input)

    # # non-square kernels and unequal stride and with padding
    model = op(16, 33, kernel_size=(3), stride=(2), padding=(4))
    execute_and_check_wrapper(model, input)

    # # non-square kernels and unequal stride and with padding and dilation
    model = op(16, 33, (3), stride=(2), padding=(4), dilation=(3))
    execute_and_check_wrapper(model, input)


@pytest.mark.parametrize("op", conv_2D)
def test_conv2D(op):

    torch.manual_seed(42)

    input = torch.randn(20, 16, 50, 10)

    # With square kernels and equal stride
    model = op(16, 33, 3, stride=2)
    execute_and_check_wrapper(model, input)

    # non-square kernels and unequal stride and with padding
    model = op(16, 33, (3, 5), stride=(2), padding=(4, 2))
    execute_and_check_wrapper(model, input)

    # non-square kernels and unequal stride and with padding and dilation
    model = op(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3))
    execute_and_check_wrapper(model, input)


@pytest.mark.parametrize("op", conv_3D)
def test_conv3D(op):
    torch.manual_seed(42)
    input = torch.randn(2, 4, 3, 5, 10)

    # With square kernels and equal stride
    model = op(4, 6, 3, stride=2)
    execute_and_check_wrapper(model, input)

    # non-square kernels and unequal stride and with padding
    model = op(4, 6, (3, 2, 2), stride=(2, 1, 1), padding=(4, 2, 0))
    execute_and_check_wrapper(model, input)

    # non-square kernels and unequal stride and with padding and dilation
    model = op(4,
               6, (3, 4, 2),
               stride=(2, 1, 1),
               padding=(4, 2, 0),
               dilation=(3, 1, 1))
    execute_and_check_wrapper(model, input)
