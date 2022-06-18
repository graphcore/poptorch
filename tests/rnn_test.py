#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import torch
import torch.nn as nn
import poptorch
import helpers


@pytest.mark.parametrize("nonlinearity", ['tanh', 'relu'])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_rnn(nonlinearity, batch_first, trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): NotImplementedError: Cannot access storage of "
            "IpuTensorImpl")
    torch.manual_seed(42)
    num_batches = 10
    sequence_length = 5
    batch_size = 8
    input_size = 4
    hidden_size = 3
    num_layers = 1

    if batch_first:
        input_shape = (batch_size, sequence_length, input_size)
    else:
        input_shape = (sequence_length, batch_size, input_size)

    inputs = [torch.randn(input_shape) for _ in range(num_batches)]
    h = torch.randn((num_layers, batch_size, hidden_size))

    rnn = nn.RNN(
        input_size,
        hidden_size,
        num_layers,
        nonlinearity=nonlinearity,
        batch_first=batch_first,
    )
    model = helpers.ModelWithWeights(rnn, inputs[0].shape, lambda x: x[0])
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    ipu_model = poptorch.trainingModel(model, options=options)

    for input in inputs:
        (out_cpu, h_cpu), _ = model((input, h))
        (out_ipu, h_ipu), _ = ipu_model((input, h))
        helpers.assert_allclose(actual=out_ipu, expected=out_cpu)
        helpers.assert_allclose(actual=h_ipu, expected=h_cpu)
        ipu_model.assert_weights_changed()
        h = h_cpu
