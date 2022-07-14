#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest
import torch
import poptorch


class SimpleModel(torch.nn.Module):
    def forward(self, log_probs, lengths):
        return poptorch.ctc_beam_search_decoder(log_probs, lengths)


@pytest.mark.parametrize("trace_model", [True, False])
def test_ctc_decoder(trace_model):
    input_size = 9
    batch_size = 3
    num_classes = 10

    torch.manual_seed(42)
    log_probs = torch.randn(input_size, batch_size, num_classes)
    lengths = torch.randint(5, input_size, (batch_size, ), dtype=torch.int)

    model = SimpleModel()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)

    result = poptorch_model(log_probs, lengths)

    # note we have no reference implementation so only the most basic
    # test is possible - relying on popart/poplibs which are validated
    # against tensorflow's implementation
    assert result[0].shape == (batch_size, 1)
    assert result[1].shape == (batch_size, 1)
    assert result[2].shape == (batch_size, 1, input_size)
