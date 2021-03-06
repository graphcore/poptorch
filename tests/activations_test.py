#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch

import pytest

# Non-linear activations (Weighted activations)
#'torch.nn.ELU', 'torch.nn.Hardshrink', 'torch.nn.Hardtanh', 'torch.nn.LeakyReLU', 'torch.nn.LogSigmoid', 'torch.nn.MultiheadAttention', 'torch.nn.MultiheadAttention.forward',
#'torch.nn.PReLU', 'torch.nn.ReLU', 'torch.nn.ReLU6', 'torch.nn.RReLU', 'torch.nn.SELU', 'torch.nn.CELU', 'torch.nn.GELU', 'torch.nn.Sigmoid', 'torch.nn.Softplus',
#'torch.nn.Softshrink', 'torch.nn.Softsign', 'torch.nn.Tanh', 'torch.nn.Tanhshrink', 'torch.nn.Threshold',

# Non-linear activations (other)
#'torch.nn.Softmin', 'torch.nn.Softmax', 'torch.nn.Softmax2d', 'torch.nn.LogSoftmax', 'torch.nn.AdaptiveLogSoftmaxWithLoss', 'torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob',
#'torch.nn.AdaptiveLogSoftmaxWithLoss.predict',

activation_functions = [
    torch.nn.ReLU, torch.nn.Tanh, torch.nn.Sigmoid, torch.nn.PReLU,
    torch.nn.SELU, torch.nn.ELU, torch.nn.GELU, torch.nn.Softmax,
    torch.nn.LogSoftmax, torch.nn.Softsign, torch.nn.LeakyReLU,
    torch.nn.Hardtanh
]


@pytest.mark.parametrize("op", activation_functions)
def test_activations(op):

    torch.manual_seed(42)

    input = torch.randn([2, 200])

    if op in (torch.nn.Softmax, torch.nn.LogSoftmax):
        model = op(dim=1)
    else:
        model = op()

    # Run on CPU.
    nativeOut = model(input)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input)

    if isinstance(model, torch.nn.GELU):
        torch.testing.assert_allclose(poptorch_out,
                                      nativeOut,
                                      rtol=0.01,
                                      atol=1e-03,
                                      equal_nan=True)
    else:
        torch.testing.assert_allclose(poptorch_out, nativeOut, equal_nan=True)


@pytest.mark.parametrize("dim", range(5))
def test_glu(dim):
    N, C, M, K, L = 2, 4, 6, 8, 10

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.glu(x, dim=dim)

    model = Model()

    torch.manual_seed(42)
    input = torch.randn(N, C, M, K, L)
    # Run on CPU.
    nativeOut = model(input)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input)

    torch.testing.assert_allclose(nativeOut, poptorch_out)


def test_logsoftmax_numerics():
    model = torch.nn.LogSoftmax(dim=1)
    x = torch.FloatTensor([[10., 100., 1000.]])
    native_out = model(x)

    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    torch.testing.assert_allclose(poptorch_out, native_out)
