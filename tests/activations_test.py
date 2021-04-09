#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import poptorch
import helpers

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
    torch.nn.Hardtanh, torch.nn.Softplus, torch.nn.Softshrink,
    torch.nn.Hardshrink, torch.nn.CELU
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
    native_out = model(input)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input)

    if isinstance(model, torch.nn.GELU):
        helpers.assert_allclose(actual=poptorch_out,
                                expected=native_out,
                                rtol=0.01,
                                atol=1e-03,
                                equal_nan=True)
    else:
        helpers.assert_allclose(actual=poptorch_out,
                                expected=native_out,
                                equal_nan=True)


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
    native_out = model(input)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input)

    helpers.assert_allclose(expected=native_out, actual=poptorch_out)


def test_logsoftmax_numerics():
    model = torch.nn.LogSoftmax(dim=1)
    x = torch.FloatTensor([[10., 100., 1000.]])
    native_out = model(x)

    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    helpers.assert_allclose(actual=poptorch_out, expected=native_out)


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
@pytest.mark.filterwarnings("ignore:Trace had nondeterministic nodes")
@pytest.mark.filterwarnings("ignore:Output nr 1. of the traced function")
def test_rrelu_training():
    opts = poptorch.Options().randomSeed(0)
    input = torch.randn([30000])

    model = torch.nn.RReLU()

    # in training negative inputs are multiplied by a random parameter
    # we'll check positive outputs and distribution of negative outputs
    native_out = model(input)
    poptorch_model = poptorch.inferenceModel(model, options=opts)
    poptorch_out = poptorch_model(input)

    ref = native_out[native_out >= 0]
    out = poptorch_out[poptorch_out >= 0]
    helpers.assert_allclose(actual=out, expected=ref)

    ref = native_out[native_out < 0]
    out = poptorch_out[poptorch_out < 0]
    for stat in [torch.mean, torch.var]:
        helpers.assert_allclose(actual=stat(out),
                                expected=stat(ref),
                                atol=0.1,
                                rtol=0.1)


def test_rrelu_inference():
    torch.manual_seed(42)
    input = torch.randn([200])

    model = torch.nn.RReLU()

    # in inference there is no randomness - check results directly
    model.eval()
    native_out = model(input)
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input)
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)
