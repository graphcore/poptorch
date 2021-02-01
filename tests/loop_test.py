#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import pytest
import poptorch


def test_loop_constant():
    class Model(torch.nn.Module):
        def forward(self, x):
            def body(x):
                return x * 2

            return poptorch.for_loop(10, body, [x])[0]

    inference_model = poptorch.inferenceModel(Model())

    x = torch.tensor([1.])

    assert inference_model(x) == pow(2, 10)


def test_loop_simple():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            def body(x):
                return x * y

            return poptorch.for_loop(10, body, [x])[0]

    inference_model = poptorch.inferenceModel(Model())

    x = torch.tensor([1.])
    y = torch.tensor([2.])

    assert inference_model(x, y) == pow(2, 10)


# TODO(T33273)
@pytest.mark.skip(reason="Popart doesn't support weights in loop oddly")
def test_loop_weights():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.layer1 = torch.nn.Linear(1, 256)
            self.layer2 = torch.nn.Conv2d(4, 1, [8, 8])

        def forward(self, x):
            def body(x):
                act = self.layer1(x)
                act = act.reshape([1, 4, 8, 8])
                act = self.layer2(act)
                return act.flatten()

            return poptorch.for_loop(10, body, [x])[0]

    inference_model = poptorch.inferenceModel(Model())

    x = torch.tensor([1.])

    inference_model(x)
