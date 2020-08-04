#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import pytest
import poptorch

# Random Sampling


# torch.rand
# Filter the following expected warnings
@pytest.mark.filterwarnings("ignore:Trace had nondeterministic nodes")
@pytest.mark.filterwarnings(
    "ignore:Output nr 1. of the traced function does not match")
def test_rand():
    class Model(torch.nn.Module):
        def forward(self):
            torch.manual_seed(42)
            return torch.rand(3, 5, 100)

    torch.manual_seed(42)
    model = Model()
    model.eval()

    # Run on CPU.
    native_out = model()

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model()

    assert native_out.size() == poptorch_out.size()

    # PRNG depends on HW implementation so we just check
    # that the distribution statistics are consistent
    stat_funs = [torch.min, torch.max, torch.mean, torch.var]

    for stat in stat_funs:
        torch.testing.assert_allclose(stat(native_out),
                                      stat(poptorch_out),
                                      atol=1e-2,
                                      rtol=0.1)
