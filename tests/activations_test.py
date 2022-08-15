#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import torch.nn as nn
import poptorch
import helpers

# Non-linear activations (Weighted activations)
# 'torch.nn.ELU', 'torch.nn.Hardshrink', 'torch.nn.Hardtanh', 'torch.nn.LeakyReLU', 'torch.nn.LogSigmoid', 'torch.nn.MultiheadAttention', 'torch.nn.MultiheadAttention.forward',
# 'torch.nn.PReLU', 'torch.nn.ReLU', 'torch.nn.ReLU6', 'torch.nn.RReLU', 'torch.nn.SELU', 'torch.nn.SiLU', 'torch.nn.CELU', 'torch.nn.GELU', 'torch.nn.Sigmoid', 'torch.nn.Softplus',
# 'torch.nn.Softshrink', 'torch.nn.Softsign', 'torch.nn.Tanh', 'torch.nn.Tanhshrink', 'torch.nn.Threshold',

# Non-linear activations (other)
# 'torch.nn.Softmin', 'torch.nn.Softmax', 'torch.nn.Softmax2d', 'torch.nn.LogSoftmax', 'torch.nn.AdaptiveLogSoftmaxWithLoss', 'torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob',
# 'torch.nn.AdaptiveLogSoftmaxWithLoss.predict',


# A version of Softplus with non default arguments
class SoftplusWithParams(nn.Softplus):
    def __init__(self):
        super().__init__(beta=5.0, threshold=4.0)


activation_functions = [
    nn.ReLU, nn.Tanh, nn.Sigmoid, nn.SELU, nn.SiLU, nn.ELU, nn.GELU,
    nn.Softmax, nn.LogSoftmax, nn.Softsign, nn.LeakyReLU, nn.Hardtanh,
    nn.Softplus, nn.Softshrink, nn.Hardshrink, nn.CELU, nn.Hardsigmoid,
    nn.Hardswish, SoftplusWithParams
]


@pytest.mark.parametrize("op", activation_functions)
@pytest.mark.parametrize("trace_model", [True, False])
def test_activations(op, trace_model):
    torch.manual_seed(42)

    input = torch.randn([2, 20])

    fn = op(dim=1) if op in (nn.Softmax, nn.LogSoftmax) else op()

    model = helpers.ModelWithWeights(fn, input.shape)
    model.train()

    # Run on CPU.
    native_out, _ = model((input, ))

    # Run on IPU.
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=options)
    poptorch_out, _ = poptorch_model((input, ))

    tol = [0.01, 1e-3] if op is nn.GELU else [1e-4, 1e-7]

    # Inference test - check outputs
    helpers.assert_allclose(actual=poptorch_out,
                            expected=native_out,
                            rtol=tol[0],
                            atol=tol[1],
                            equal_nan=True)

    poptorch_model.assert_weights_changed()


@pytest.mark.parametrize("trace_model", [True, False])
@pytest.mark.parametrize("input", [
    torch.randn((4, )),
    torch.randn((2, 2)),
    torch.randn((2, 8, 16)),
    torch.randn((2, 8, 32, 32))
])
def test_prelu(trace_model, input):
    num_channels = input.shape[1] if input.dim() >= 2 else 1
    model = nn.PReLU(num_channels)

    # Run on CPU.
    native_out = model(input)

    # Run on IPU.
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options=options)
    poptorch_out = poptorch_model(input)

    helpers.assert_allclose(actual=poptorch_out,
                            expected=native_out,
                            rtol=1e-4,
                            atol=1e-7,
                            equal_nan=True)


@pytest.mark.parametrize("dim", range(5))
@pytest.mark.parametrize("trace_model", [True, False])
def test_glu(dim, trace_model):
    torch.manual_seed(42)
    N, C, M, K, L = 2, 4, 6, 8, 10

    input = torch.randn(N, C, M, K, L)
    model = helpers.ModelWithWeights(nn.GLU(dim=dim), input.shape)

    # Run on CPU.
    native_out, _ = model((input, ))

    # Run on IPU.
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=options)
    poptorch_out, _ = poptorch_model((input, ))

    # Inference test - check outputs
    helpers.assert_allclose(expected=native_out, actual=poptorch_out)

    # Training test - check weights have changed
    poptorch_model.assert_weights_changed()


@pytest.mark.parametrize("op", activation_functions)
@pytest.mark.parametrize("trace_model", [True, False])
def test_activation_numerics(op, trace_model):
    enable_exceptions = True
    if op in (nn.SELU, nn.ELU, nn.CELU):
        # These activations rely on exponentials that will overflow
        # but saturate to a linear function in the range where x >> 0
        enable_exceptions = False

    model = op(dim=1) if op in (nn.Softmax, nn.LogSoftmax) else op()
    x = torch.FloatTensor([[10., 100., 1000.]])
    native_out = model(x)

    options = poptorch.Options()
    options.Precision.enableFloatingPointExceptions(enable_exceptions)
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options=options)
    poptorch_out = poptorch_model(x)

    helpers.assert_allclose(actual=poptorch_out, expected=native_out)


@pytest.mark.ipuHardwareRequired
@pytest.mark.filterwarnings("ignore:Trace had nondeterministic nodes")
@pytest.mark.filterwarnings("ignore:Output nr 1. of the traced function")
@pytest.mark.filterwarnings("ignore:Output nr 2. of the traced function")
@pytest.mark.parametrize("trace_model", [True, False])
def test_rrelu_training(trace_model):
    opts = poptorch.Options().randomSeed(0)
    opts.Jit.traceModel(trace_model)

    input = torch.randn([3000])

    model = helpers.ModelWithWeights(nn.RReLU(), input.shape)

    # in training negative inputs are multiplied by a random parameter
    # we'll check positive outputs and distribution of negative outputs
    native_out, _ = model((input, ))
    poptorch_model = poptorch.trainingModel(model, options=opts)
    poptorch_out, _ = poptorch_model((input, ))

    ref = native_out[native_out >= 0]
    out = poptorch_out[poptorch_out >= 0]
    helpers.assert_allclose(actual=out, expected=ref)

    ref = native_out[native_out < 0]
    out = poptorch_out[poptorch_out < 0]
    # Inference test - check outputs
    for stat in [torch.mean, torch.var]:
        helpers.assert_allclose(actual=stat(out),
                                expected=stat(ref),
                                atol=0.1,
                                rtol=0.1)

    # Training test - check weights have changed
    poptorch_model.assert_weights_changed()


@pytest.mark.parametrize("trace_model", [True, False])
def test_rrelu_inference(trace_model):
    torch.manual_seed(42)
    input = torch.randn([200])

    model = nn.RReLU()

    # in inference there is no randomness - check results directly
    model.eval()
    native_out = model(input)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options=options)
    poptorch_out = poptorch_model(input)
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)
