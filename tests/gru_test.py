#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import poptorch
import helpers


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
def test_gru_inference(bias, batch_first):
    length = 1
    batches = 3
    input_size = 5
    hidden_size = 7

    layers = 1
    directions = 1

    torch.manual_seed(42)
    if batch_first:
        inp = torch.randn(batches, length, input_size)
    else:
        inp = torch.randn(length, batches, input_size)
    h0 = torch.randn(layers * directions, batches, hidden_size)

    model = torch.nn.GRU(input_size,
                         hidden_size,
                         bias=bias,
                         batch_first=batch_first)

    poptorch_model = poptorch.inferenceModel(model)

    (native_out, native_hn) = model(inp, h0)
    (poptorch_out, poptorch_hn) = poptorch_model(inp, h0)

    helpers.assert_allclose(actual=poptorch_out, expected=native_out)
    helpers.assert_allclose(actual=poptorch_hn, expected=native_hn)
