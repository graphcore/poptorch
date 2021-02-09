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


def test_loop_multiple_inputs():
    class Model(torch.nn.Module):
        def forward(self, x, y, z, w):
            def body(x, y, z, w):
                return x * y, y + z, x * w, w + 1

            return poptorch.for_loop(10, body, [x, y, z, w])

    inference_model = poptorch.inferenceModel(Model())

    x = torch.tensor([0.1])
    y = torch.tensor([0.2])
    z = torch.tensor([0.3])
    w = torch.tensor([0.4])

    out = inference_model(x, y, z, w)

    # Check by running equiv on host.
    x = torch.tensor([0.1])
    y = torch.tensor([0.2])
    z = torch.tensor([0.3])
    w = torch.tensor([0.4])

    for _ in range(0, 10):
        _z = x * w
        x *= y
        y += z
        w = w + 1
        z = _z

    for host, ipu in zip([x, y, z, w], out):
        assert host == ipu


def test_loop_non_tensor_in():
    class Model(torch.nn.Module):
        def forward(self, x, _):
            def body(x, y):
                return x * y, y + 1

            return poptorch.for_loop(10, body, [x, 5])

    inference_model = poptorch.inferenceModel(Model())

    x = torch.tensor([1.])
    y = torch.tensor([2.])

    msg = "(Object contained in list at index 1 is not torch.tensor)"
    with pytest.raises(ValueError, match=msg):
        inference_model(x, y)


def test_loop_non_list_in():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            def body(x):
                return x * y

            return poptorch.for_loop(10, body, x)

    inference_model = poptorch.inferenceModel(Model())

    x = torch.tensor([1.])
    y = torch.tensor([2.])

    msg = "(Object is not list)"
    with pytest.raises(ValueError, match=msg):
        inference_model(x, y)


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
