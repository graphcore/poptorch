#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
import pytest
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

    assert torch.equal(nativeOut, poptorch_out.long())


def test_one_hot_invalid():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.one_hot(x, num_classes=-1)

    input = torch.randint(high=10, size=[10])
    model = Model()

    msg = "OneHot num classes must be specified and must be constant."
    # Run on IPU.
    with pytest.raises(RuntimeError, match=msg):
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
    assert torch.equal(nativeOut, poptorch_out)
