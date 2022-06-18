#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import pytest
import poptorch
import helpers


@pytest.mark.parametrize("norm", {1., 2., 3., 4.})
@pytest.mark.parametrize("trace_model", [True, False])
def test_pairwise_distance(norm, trace_model):
    torch.manual_seed(42)

    size = [10, 5]
    input1 = torch.randn(size)
    input2 = torch.randn(size)
    shape = input1.shape

    model = helpers.ModelWithWeights(torch.nn.PairwiseDistance(norm), shape)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=options)

    # Run on CPU
    native_out, _ = model((input1, input2))

    # Run on IPU
    poptorch_out, _ = poptorch_model((input1, input2))

    # Inference test - check outputs
    helpers.assert_allclose(expected=native_out, actual=poptorch_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


@pytest.mark.parametrize("dim", {0, 1})
@pytest.mark.parametrize("trace_model", [True, False])
def test_cosine_similarity(dim, trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::clamp_min.out")
    torch.manual_seed(42)

    size = [10, 5]
    input1 = torch.randn(size)
    input2 = torch.randn(size)
    shape = input1.shape

    model = helpers.ModelWithWeights(torch.nn.CosineSimilarity(dim), shape)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=options)

    # Run on CPU
    native_out, _ = model((input1, input2))

    # Run on IPU
    poptorch_out, _ = poptorch_model((input1, input2))

    # Inference test - check outputs
    helpers.assert_allclose(expected=native_out, actual=poptorch_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()
