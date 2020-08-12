#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import ctypes
import platform
import torch
import torch.nn as nn
import poptorch

if platform.system() == "Darwin":
    myso = os.path.join(os.getcwd(), "custom_ops/libcustom_cube_op.dylib")
else:
    myso = os.path.join(os.getcwd(), "custom_ops/libcustom_cube_op.so")

myop = ctypes.cdll.LoadLibrary(myso)


def test_inference():
    class BasicNetwork(nn.Module):
        def forward(self, x, bias):
            x, y = poptorch.custom_op([x, bias],
                                      "Cube",
                                      "com.acme",
                                      1,
                                      example_outputs=[x, x])
            return x, y

    model = BasicNetwork()

    x = torch.full((1, 8), 2)
    bias = torch.full((1, 8), 4)

    inference_model = poptorch.inferenceModel(model)
    out = inference_model(x, bias)

    print(out)
    torch.testing.assert_allclose(out[0], 12.0)
    torch.testing.assert_allclose(out[1], 8.0)


def test_training():
    class TrainingNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = torch.nn.Linear(100, 100)
            self.softmax = nn.Softmax(1)

        def forward(self, x, bias):
            x, y = poptorch.custom_op([x, bias],
                                      "Cube",
                                      "com.acme",
                                      1,
                                      example_outputs=[x, x])
            x = self.ln(x)
            return self.softmax(x), y

    model = TrainingNetwork()

    x = torch.rand((1, 100))
    bias = torch.full((1, 100), 2)

    y = torch.full([1], 42, dtype=torch.long)

    def custom_loss(model_out, labels):
        l1 = torch.nn.functional.nll_loss(model_out[0], labels)
        # Popart errors if this is unused.
        l2 = torch.sum(model_out[1]) * 0.0001

        return l1 + l2

    training = poptorch.trainingModel(model, loss=custom_loss)

    for _ in range(0, 100):
        x = torch.rand((1, 100))
        out, _ = training((x, bias), y)

    assert torch.argmax(out[0]) == 42
