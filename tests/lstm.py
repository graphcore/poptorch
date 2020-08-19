#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import poptorch
import poptorch.testing
import helpers


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
    out = lstm(inputs, hidden)
    ipuOut = ipuLstm(inputs, hidden)
    assert poptorch.testing.allclose(out, ipuOut)


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
    out = lstm(inputs, hidden)
    ipuOut = ipuLstm(inputs, hidden)
    assert poptorch.testing.allclose(out, ipuOut)


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
    out = lstm(inputs, hidden)
    ipuOut = ipuLstm(inputs, hidden)
    assert poptorch.testing.allclose(out, ipuOut)


def test_lstm_fc():
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, classes):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size,
                                hidden_size,
                                num_layers=1,
                                bias=True)
            self.fc = nn.Linear(hidden_size, classes, bias=False)

        def forward(self, x):
            h1, _ = self.lstm(x)
            h2 = h1[-1, :, :]
            h3 = self.fc(h2)
            return h3

    torch.manual_seed(42)
    batch_size = 2
    input_size = 5
    classes = 3
    lstm = LSTMModel(input_size=input_size, hidden_size=3, classes=classes)

    ipuLstm = poptorch.inferenceModel(lstm)
    input = torch.randn(1, batch_size, input_size)
    out = lstm(input)
    ipuOut = ipuLstm(input)
    assert poptorch.testing.allclose(out, ipuOut)


def test_lstm_fc_training():
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, classes):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size,
                                hidden_size,
                                num_layers=1,
                                bias=True)
            self.fc = nn.Linear(hidden_size, classes, bias=False)

        def forward(self, x):
            h1, _ = self.lstm(x)
            h2 = h1[-1, :, :]
            h3 = self.fc(h2)
            return h3

    torch.manual_seed(42)
    batch_size = 2
    input_size = 5
    classes = 3
    lstm = LSTMModel(input_size=input_size, hidden_size=3, classes=classes)

    ipuLstm = helpers.trainingModelWithLoss(lstm, loss=nn.CrossEntropyLoss())
    input = torch.randn(1, batch_size, input_size)
    label = torch.tensor([1, 2])
    ipuLstm(input, label.long())
