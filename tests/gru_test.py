#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import poptorch
import helpers


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_gru(bias, batch_first, trace_model):
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

    op = torch.nn.GRU(input_size,
                      hidden_size,
                      bias=bias,
                      batch_first=batch_first)

    out_fn = lambda x: x[0]
    model = helpers.ModelWithWeights(op, inp.shape, out_fn)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)

    poptorch_model = poptorch.trainingModel(model, options=options)

    (native_out, native_hn), _ = model((inp, h0))
    (poptorch_out, poptorch_hn), _ = poptorch_model((inp, h0))

    # Inference test - check outputs
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)
    helpers.assert_allclose(actual=poptorch_hn, expected=native_hn)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()
