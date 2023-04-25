#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import pytest
import helpers
import poptorch


@pytest.mark.parametrize("norm", {1., 2., 3., 4.})
def test_pairwise_distance(norm):
    torch.manual_seed(42)

    size = [10, 5]
    input1 = torch.randn(size)
    input2 = torch.randn(size)
    shape = input1.shape

    model = helpers.ModelWithWeights(torch.nn.PairwiseDistance(norm), shape)
    poptorch_model = poptorch.trainingModel(model)

    # Run on CPU
    native_out, _ = model((input1, input2))

    # Run on IPU
    poptorch_out, _ = poptorch_model((input1, input2))

    # Inference test - check outputs
    helpers.assert_allclose(expected=native_out, actual=poptorch_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


@pytest.mark.parametrize("dim", {0, 1})
def test_cosine_similarity(dim):
    torch.manual_seed(42)

    size = [10, 5]
    input1 = torch.randn(size)
    input2 = torch.randn(size)
    shape = input1.shape

    model = helpers.ModelWithWeights(torch.nn.CosineSimilarity(dim), shape)
    poptorch_model = poptorch.trainingModel(model)

    # Run on CPU
    native_out, _ = model((input1, input2))

    # Run on IPU
    poptorch_out, _ = poptorch_model((input1, input2))

    # Inference test - check outputs
    helpers.assert_allclose(expected=native_out, actual=poptorch_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


@pytest.mark.parametrize("input_shapes",
                         (((3, 2), (2, 2)), ((3, 2, 3), (3, 10, 3)),
                          ((3, 5, 2, 7), (5, 11, 7)), ((3, 5, 1, 2, 7),
                                                       (3, 1, 10, 11, 7))))
@pytest.mark.parametrize("p", (2, 3))
def test_cdist(input_shapes, p):
    a_shape, b_shape = input_shapes

    torch.manual_seed(42)

    class Cdist(torch.nn.Module):
        def __init__(self, p, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.p = p

        def forward(self, x, y):
            return torch.cdist(x, y, self.p)

    a = torch.rand(*a_shape)
    b = torch.rand(*b_shape)

    model = helpers.ModelWithWeights(Cdist(p), a.shape)

    # Run on CPU
    native_out, _ = model((a, b))

    poptorch_model = poptorch.trainingModel(model)

    # Run on IPU
    poptorch_out, _ = poptorch_model((a, b))

    # Inference test - check outputs
    helpers.assert_allclose(expected=native_out, actual=poptorch_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()
