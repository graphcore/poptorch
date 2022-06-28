#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import torch.nn as nn
import poptorch
import helpers


@pytest.mark.parametrize("trace_model", [True, False])
def test_lstm(trace_model):
    torch.manual_seed(42)
    lstm = nn.LSTM(3, 3)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    ipuLstm = poptorch.inferenceModel(lstm, options)
    inputs = [torch.randn(1, 3) for _ in range(5)]
    # initialize the hidden state.
    hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
    for i in inputs:
        # Step through the sequence one element at a time.
        # after each step, hidden contains the hidden state.
        out, newHidden = lstm(i.view(1, 1, -1), hidden)
        ipuOut, ipuHidden = ipuLstm(i.view(1, 1, -1), hidden)
        helpers.assert_allclose(expected=newHidden[0], actual=ipuHidden[0])
        helpers.assert_allclose(expected=newHidden[1], actual=ipuHidden[1])
        helpers.assert_allclose(expected=out, actual=ipuOut)
        hidden = newHidden


@pytest.mark.parametrize("trace_model", [True, False])
def test_lstm2(trace_model):
    torch.manual_seed(42)
    numHidden = 5
    inputSize = 3
    lstm = nn.LSTM(3, numHidden)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    ipuLstm = poptorch.inferenceModel(lstm, options)
    inputs = [torch.randn(1, inputSize) for _ in range(5)]
    # Add the extra 2nd dimension
    inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    hidden = (torch.randn(1, 1, numHidden), torch.randn(1, 1, numHidden))
    out, newHidden = lstm(inputs, hidden)
    ipuOut, ipuHidden = ipuLstm(inputs, hidden)
    helpers.assert_allclose(expected=newHidden[0], actual=ipuHidden[0])
    helpers.assert_allclose(expected=newHidden[1], actual=ipuHidden[1])
    helpers.assert_allclose(expected=out, actual=ipuOut)


@pytest.mark.parametrize("trace_model", [True, False])
def test_lstm_twice(trace_model):
    torch.manual_seed(42)
    numHidden = 5
    inputSize = 3
    lstm = nn.LSTM(3, numHidden)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    ipuLstm = poptorch.inferenceModel(lstm, options)
    inputs = [torch.randn(1, inputSize) for _ in range(5)]
    # Add the extra 2nd dimension
    inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    hidden = (torch.randn(1, 1, numHidden), torch.randn(1, 1, numHidden))
    out, newHidden = lstm(inputs, hidden)
    ipuOut, ipuHidden = ipuLstm(inputs, hidden)
    helpers.assert_allclose(expected=newHidden[0], actual=ipuHidden[0])
    helpers.assert_allclose(expected=newHidden[1], actual=ipuHidden[1])
    helpers.assert_allclose(expected=out, actual=ipuOut)

    out, newHidden = lstm(inputs, hidden)
    ipuOut2, ipuHidden2 = ipuLstm(inputs, hidden)
    helpers.assert_allclose(expected=newHidden[0], actual=ipuHidden2[0])
    helpers.assert_allclose(expected=newHidden[1], actual=ipuHidden2[1])
    helpers.assert_allclose(expected=out, actual=ipuOut2)
    helpers.assert_allclose(expected=ipuOut, actual=ipuOut2)


@pytest.mark.parametrize("trace_model", [True, False])
def test_lstm_batch_first(trace_model):
    torch.manual_seed(42)
    numHidden = 5
    inputSize = 3
    lstm = nn.LSTM(3, numHidden, batch_first=True)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    ipuLstm = poptorch.inferenceModel(lstm, options)
    inputs = [torch.randn(1, inputSize) for _ in range(5)]
    # Add the extra 2nd dimension
    inputs = torch.cat(inputs).view(1, len(inputs), -1)
    hidden = (torch.randn(1, 1, numHidden), torch.randn(1, 1, numHidden))
    out, newHidden = lstm(inputs, hidden)
    ipuOut, ipuHidden = ipuLstm(inputs, hidden)
    helpers.assert_allclose(expected=newHidden[0], actual=ipuHidden[0])
    helpers.assert_allclose(expected=newHidden[1], actual=ipuHidden[1])
    helpers.assert_allclose(expected=out, actual=ipuOut)


@pytest.mark.parametrize("trace_model", [True, False])
def test_lstm_batched(trace_model):
    torch.manual_seed(42)
    numHidden = 5
    inputSize = 3
    batch = 4
    lstm = nn.LSTM(3, numHidden)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    ipuLstm = poptorch.inferenceModel(lstm, options)
    inputs = [torch.randn(batch, inputSize) for _ in range(5)]
    # Add the extra 2nd dimension
    inputs = torch.cat(inputs).view(len(inputs), batch, -1)
    print(inputs.shape)
    hidden = (torch.randn(1, batch,
                          numHidden), torch.randn(1, batch, numHidden))
    out, newHidden = lstm(inputs, hidden)
    ipuOut, ipuHidden = ipuLstm(inputs, hidden)
    helpers.assert_allclose(expected=newHidden[0], actual=ipuHidden[0])
    helpers.assert_allclose(expected=newHidden[1], actual=ipuHidden[1])
    helpers.assert_allclose(expected=out, actual=ipuOut)


@pytest.mark.parametrize("trace_model", [True, False])
def test_lstm_batched_batch_first(trace_model):
    torch.manual_seed(42)
    numHidden = 5
    inputSize = 3
    batch = 4
    lstm = nn.LSTM(3, numHidden, batch_first=True)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    ipuLstm = poptorch.inferenceModel(lstm, options)
    inputs = [torch.randn(batch, inputSize) for _ in range(5)]
    # Add the extra 2nd dimension
    inputs = torch.cat(inputs).view(batch, len(inputs), -1)
    hidden = (torch.randn(1, batch,
                          numHidden), torch.randn(1, batch, numHidden))
    out, newHidden = lstm(inputs, hidden)
    ipuOut, ipuHidden = ipuLstm(inputs, hidden)
    helpers.assert_allclose(expected=newHidden[0], actual=ipuHidden[0])
    helpers.assert_allclose(expected=newHidden[1], actual=ipuHidden[1])
    helpers.assert_allclose(expected=out, actual=ipuOut)


@pytest.mark.parametrize("trace_model", [True, False])
def test_lstm_fc(trace_model):
    if not trace_model:
        pytest.skip("TODO(T57195): Could not find cpu tensor")

    torch.manual_seed(42)

    batch_size = 2
    input_size = 5

    op = nn.LSTM(input_size, hidden_size=3, num_layers=1, bias=True)

    input = torch.randn(1, batch_size, input_size)
    out_fn = lambda x: x[0]
    model = helpers.ModelWithWeights(op, input.shape, out_fn)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=options)

    (native_out, (native_hn, native_cn)), _ = model((input, ))
    (poptorch_out, (poptorch_hn, poptorch_cn)), _ = poptorch_model((input, ))

    # Inference test - check outputs
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)
    helpers.assert_allclose(actual=poptorch_hn, expected=native_hn)
    helpers.assert_allclose(actual=poptorch_cn, expected=native_cn)

    # Training test - check weights have changed
    poptorch_model.assert_weights_changed()
