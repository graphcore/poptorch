#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import pytest
import poptorch


@pytest.mark.parametrize("norm", {1., 2., 3., 4.})
def test_pairwise_distance(norm):
    torch.manual_seed(42)

    input1 = torch.randn(10, 5)
    input2 = torch.randn(10, 5)

    model = torch.nn.PairwiseDistance(norm)
    poptorch_model = poptorch.inferenceModel(model)

    # Run on CPU
    native_out = model(input1, input2)

    # Run on IPU
    poptorch_out = poptorch_model(input1, input2)

    torch.testing.assert_allclose(native_out, poptorch_out)
