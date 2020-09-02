#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch
import pytest


class ConstantBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('stuff', torch.tensor([1, 2, 3],
                                                   dtype=torch.int32))

    def forward(self, x):
        new_stuff = 1.0 + self.stuff
        return torch.sum(x + new_stuff)


def test_constant_buffer():
    model = ConstantBuffer()

    poptorch_model = poptorch.inferenceModel(model)
    assert poptorch_model(torch.tensor([2])) == 15
    #Unfortnately any future tests will fail due to popart implacing issue


@pytest.mark.xfail(strict=True)
def test_constant_buffer_repeat():
    model = ConstantBuffer()

    poptorch_model = poptorch.inferenceModel(model)
    assert poptorch_model(torch.tensor([2])) == 15
    assert poptorch_model(torch.tensor([2])) == 15
    #Expect this to fail for now
