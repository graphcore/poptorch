#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
import pytest
import helpers
import poptorch


def test_one_hot():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.one_hot(x, num_classes=10)

    input = torch.randint(high=10, size=[10, 5, 4])
    model = Model()

    # Run on CPU.
    nativeOut = model(input)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input)

    helpers.assert_allequal(actual=poptorch_out.long(), expected=nativeOut)


def test_one_hot_invalid():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.one_hot(x, num_classes=-1)

    input = torch.randint(high=10, size=[10])
    model = Model()

    msg = "OneHot num classes must be specified and must be constant."
    # Run on IPU.
    with pytest.raises(poptorch.Error, match=msg):
        poptorch_model = poptorch.inferenceModel(model)
        poptorch_model(input)


def test_one_hot_casted():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = torch.nn.functional.one_hot(x, num_classes=10)
            return x.half()

    input = torch.randint(high=10, size=[10, 5, 4])
    model = Model()

    # Run on CPU.
    nativeOut = model(input)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input)

    assert poptorch_out.dtype == torch.half
    helpers.assert_allequal(actual=poptorch_out, expected=nativeOut)


@pytest.mark.parametrize("in_features,out_features", [(8, 7), (7, 6), (6, 5)])
def test_linear(in_features, out_features):
    class Model(torch.nn.Module):
        weight: torch.Tensor
        bias: torch.Tensor

        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.weight = torch.nn.parameter.Parameter(
                torch.ones((out_features, in_features), dtype=torch.float))
            self.bias = torch.nn.parameter.Parameter(torch.ones(out_features))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight, self.bias)

    input = torch.arange(out_features * in_features,
                         dtype=torch.float).reshape(out_features, in_features)
    model = Model(in_features=in_features, out_features=out_features)

    # Run on CPU.
    native_out = model(input)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input)

    assert poptorch_out.dtype == torch.float
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)
