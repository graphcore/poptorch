#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import poptorch
import poptorch.testing


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
        assert poptorch.testing.allclose(newHidden, ipuHidden)
        torch.testing.assert_allclose(out, ipuOut)
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
    out = lstm(inputs, hidden)
    ipuOut = ipuLstm(inputs, hidden)
    ipuLstm.copyWeightsToHost()
    assert poptorch.testing.allclose(out, ipuOut)


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
    out = lstm(inputs, hidden)
    ipuOut = ipuLstm(inputs, hidden)
    assert poptorch.testing.allclose(out, ipuOut)
    out = lstm(inputs, hidden)
    ipuOut2 = ipuLstm(inputs, hidden)
    assert poptorch.testing.allclose(out, ipuOut2)
    assert poptorch.testing.allclose(ipuOut, ipuOut2)
