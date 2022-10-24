#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import copy
import torch
from torch import nn
import pytest
import helpers
from poptorch.experimental import IPUContext


@pytest.mark.parametrize("lstm_config", [
    {
        'batch_size': 2,
        'batch_first': False,
        'sequence_length': 5,
        'input_size': 3,
        'hidden_size': 4
    },
    {
        'batch_size': 6,
        'batch_first': False,
        'sequence_length': 5,
        'input_size': 3,
        'hidden_size': 4
    },
    {
        'batch_size': 2,
        'batch_first': True,
        'sequence_length': 5,
        'input_size': 3,
        'hidden_size': 4
    },
    {
        'batch_size': 2,
        'batch_first': False,
        'sequence_length': 6,
        'input_size': 3,
        'hidden_size': 4
    },
    {
        'batch_size': 2,
        'batch_first': False,
        'sequence_length': 5,
        'input_size': 6,
        'hidden_size': 4
    },
    {
        'batch_size': 2,
        'batch_first': False,
        'sequence_length': 5,
        'input_size': 3,
        'hidden_size': 6
    },
    {
        'batch_size': 1,
        'batch_first': False,
        'sequence_length': 5,
        'input_size': 3,
        'hidden_size': 4
    },
])
def test_lstm(lstm_config):
    num_layers = 1

    torch.manual_seed(42)

    if lstm_config['batch_size'] == "1":
        input_shape = [
            lstm_config['sequence_length'], lstm_config['input_size']
        ]
        output_shape = [
            lstm_config['sequence_length'], lstm_config['hidden_size']
        ]
        pytest.skip()  # TODO(T57255) unbatched input supported in torch 1.12

    if lstm_config['batch_first']:
        input_shape = [
            lstm_config['batch_size'], lstm_config['sequence_length'],
            lstm_config['input_size']
        ]
        output_shape = [
            lstm_config['batch_size'], lstm_config['sequence_length'],
            lstm_config['hidden_size']
        ]
    else:
        input_shape = [
            lstm_config['sequence_length'], lstm_config['batch_size'],
            lstm_config['input_size']
        ]
        output_shape = [
            lstm_config['sequence_length'], lstm_config['batch_size'],
            lstm_config['hidden_size']
        ]

    input_t = torch.randn(input_shape)
    target_t = torch.randn(output_shape)

    ipu_lstm = nn.LSTM(input_size=lstm_config['input_size'],
                       hidden_size=lstm_config['hidden_size'],
                       num_layers=num_layers,
                       batch_first=lstm_config['batch_first'])

    cpu_lstm = copy.deepcopy(ipu_lstm)

    def cpu_step(lstm, x1, x2):
        x1.requires_grad = True
        x1.retain_grad()
        out, (h_n, c_n) = lstm(x1)
        loss = torch.nn.functional.mse_loss(out, x2)
        loss.backward()
        ret = [out, h_n, c_n]

        # For now, grad is None, in future return these
        #  x1.grad, lstm.weight_ih_l0.grad,
        #        lstm.weight_hh_l0.grad, lstm.bias_ih_l0.grad,
        #        lstm.bias_hh_l0.grad
        return ret

    ipu_result = IPUContext(cpu_step, model=ipu_lstm)(ipu_lstm, input_t,
                                                      target_t)

    cpu_result = cpu_step(cpu_lstm, input_t, target_t)

    # Test outputs and gradients
    for cpu, ipu in zip(cpu_result, ipu_result):
        helpers.assert_allclose(actual=ipu,
                                expected=cpu)  #, rtol=1e-3, atol=1e-3)
