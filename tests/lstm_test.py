#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import helpers
import poptorch


def test_lstm():
    torch.manual_seed(42)
    lstm = nn.LSTM(3, 3)
    ipuLstm = poptorch.inferenceModel(lstm)
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


def test_lstm2():
    torch.manual_seed(42)
    numHidden = 5
    inputSize = 3
    lstm = nn.LSTM(3, numHidden)
    ipuLstm = poptorch.inferenceModel(lstm)
    inputs = [torch.randn(1, inputSize) for _ in range(5)]
    # Add the extra 2nd dimension
    inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    hidden = (torch.randn(1, 1, numHidden), torch.randn(1, 1, numHidden))
    out, newHidden = lstm(inputs, hidden)
    ipuOut, ipuHidden = ipuLstm(inputs, hidden)
    helpers.assert_allclose(expected=newHidden[0], actual=ipuHidden[0])
    helpers.assert_allclose(expected=newHidden[1], actual=ipuHidden[1])
    helpers.assert_allclose(expected=out, actual=ipuOut)


def test_lstm_twice():
    torch.manual_seed(42)
    numHidden = 5
    inputSize = 3
    lstm = nn.LSTM(3, numHidden)
    ipuLstm = poptorch.inferenceModel(lstm)
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


def test_lstm_batch_first():
    torch.manual_seed(42)
    numHidden = 5
    inputSize = 3
    lstm = nn.LSTM(3, numHidden, batch_first=True)
    ipuLstm = poptorch.inferenceModel(lstm)
    inputs = [torch.randn(1, inputSize) for _ in range(5)]
    # Add the extra 2nd dimension
    inputs = torch.cat(inputs).view(1, len(inputs), -1)
    hidden = (torch.randn(1, 1, numHidden), torch.randn(1, 1, numHidden))
    out, newHidden = lstm(inputs, hidden)
    ipuOut, ipuHidden = ipuLstm(inputs, hidden)
    helpers.assert_allclose(expected=newHidden[0], actual=ipuHidden[0])
    helpers.assert_allclose(expected=newHidden[1], actual=ipuHidden[1])
    helpers.assert_allclose(expected=out, actual=ipuOut)


def test_lstm_batched():
    torch.manual_seed(42)
    numHidden = 5
    inputSize = 3
    batch = 4
    lstm = nn.LSTM(3, numHidden)
    ipuLstm = poptorch.inferenceModel(lstm)
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


def test_lstm_batched_batch_first():
    torch.manual_seed(42)
    numHidden = 5
    inputSize = 3
    batch = 4
    lstm = nn.LSTM(3, numHidden, batch_first=True)
    ipuLstm = poptorch.inferenceModel(lstm)
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


def test_lstm_fc():
    torch.manual_seed(42)

    batch_size = 2
    input_size = 5

    op = nn.LSTM(input_size, hidden_size=3, num_layers=1, bias=True)

    input = torch.randn(1, batch_size, input_size)
    out_fn = lambda x: x[0]
    model = helpers.ModelWithWeights(op, input.shape, out_fn)

    poptorch_model = poptorch.trainingModel(model)

    (native_out, (native_hn, native_cn)), _ = model((input, ))
    (poptorch_out, (poptorch_hn, poptorch_cn)), _ = poptorch_model((input, ))

    # Inference test - check outputs
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)
    helpers.assert_allclose(actual=poptorch_hn, expected=native_hn)
    helpers.assert_allclose(actual=poptorch_cn, expected=native_cn)

    # Training test - check weights have changed
    poptorch_model.assert_weights_changed()
