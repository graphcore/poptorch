#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import ctypes
import pathlib
import torch
import torch.nn as nn
import helpers
import poptorch

#loading_library_start
myso = list(pathlib.Path("tests").rglob("libcustom_cube_op.*"))
assert myso, "Failed to find libcustom_cube_op"
myop = ctypes.cdll.LoadLibrary(myso[0])

#loading_library_end

myso = list(pathlib.Path("tests").rglob("libcustom_leaky_relu_op.*"))
assert myso, "Failed to find libcustom_leaky_relu_op"
myop = ctypes.cdll.LoadLibrary(myso[0])


#inference_start
def test_inference():
    class BasicNetwork(nn.Module):
        def forward(self, x, bias):
            x, y = poptorch.custom_op([x, bias],
                                      "Cube",
                                      "com.acme",
                                      1,
                                      example_outputs=[x, x])
            return x, y

    #inference_end

    model = BasicNetwork()

    x = torch.full((1, 8), 2.0)
    bias = torch.full((1, 8), 4.0)

    inference_model = poptorch.inferenceModel(model)
    out = inference_model(x, bias)

    expected = (torch.full((1, 8), 12.0), torch.full((1, 8), 8.0))
    helpers.assert_allclose(actual=out[0], expected=expected[0])
    helpers.assert_allclose(actual=out[1], expected=expected[1])


def test_training():
    class TrainingNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = torch.nn.Linear(100, 100)
            self.softmax = nn.Softmax(1)

        def forward(self, t):
            x = t[0]
            bias = t[1]
            x, y = poptorch.custom_op([x, bias],
                                      "Cube",
                                      "com.acme",
                                      1,
                                      example_outputs=[x, x])
            x = self.ln(x)
            return self.softmax(x), y

    model = TrainingNetwork()

    x = torch.rand((1, 100))
    bias = torch.full((1, 100), 2.0)

    y = torch.full([1], 42, dtype=torch.long)

    def custom_loss(model_out, labels):
        l1 = torch.nn.functional.nll_loss(model_out[0], labels)
        # Popart errors if this is unused.
        l2 = torch.sum(model_out[1]) * 0.0001

        return l1 + l2

    training = helpers.trainingModelWithLoss(model, custom_loss)

    for _ in range(0, 100):
        x = torch.rand((1, 100))
        out, _ = training((x, bias), y)

    assert torch.argmax(out[0]) == 42


# Check that the custom op not only trains but also propagates the gradient backwards.
def test_training_both_sides():
    class TrainingNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = torch.nn.Linear(100, 100)
            self.ln2 = torch.nn.Linear(100, 100)
            self.softmax = nn.Softmax(1)

        def forward(self, t):
            x = self.ln1(t[0])
            bias = t[1]
            x, y = poptorch.custom_op([x, bias],
                                      "Cube",
                                      "com.acme",
                                      1,
                                      example_outputs=[x, x])
            x = self.ln2(x)
            return self.softmax(x), y

    model = TrainingNetwork()

    x = torch.rand((1, 100))
    bias = torch.full((1, 100), 2.0)

    y = torch.full([1], 42, dtype=torch.long)

    def custom_loss(model_out, labels):
        l1 = torch.nn.functional.nll_loss(model_out[0], labels)
        # Popart errors if this is unused.
        l2 = torch.sum(model_out[1]) * 0.0001

        return l1 + l2

    weights_before = model.ln1.weight.clone()

    training = helpers.trainingModelWithLoss(model, custom_loss)

    for _ in range(0, 100):
        x = torch.rand((1, 100))
        out, _ = training((x, bias), y)

    assert not torch.allclose(weights_before, model.ln1.weight)

    assert torch.argmax(out[0]) == 42


def test_inference_with_an_attribute():
    #inference_with_attribute_start
    class Model(torch.nn.Module):
        def forward(self, x):
            x = poptorch.custom_op([x],
                                   "LeakyRelu",
                                   "com.acme",
                                   1,
                                   example_outputs=[x],
                                   attributes={"alpha": 0.02})
            return x[0]

    #inference_with_attribute_end

    model = Model()

    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])

    inference_model = poptorch.inferenceModel(model)
    out = inference_model(x)

    helpers.assert_allclose(actual=out,
                            expected=torch.tensor(
                                [-0.02, -0.01, 0.0, 0.5, 1.0]))
