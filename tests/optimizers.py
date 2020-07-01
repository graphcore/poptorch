#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import poptorch
import torch.optim as optim


def test_SGD():
    torch.manual_seed(42)

    reductions = ["mean", "sum"]

    model = torch.nn.Linear(10, 10)

    # "Train" with learning rate of zero and check the loss remains the same.
    optimizer = optim.SGD(model.parameters(), lr=0.00)

    opts = poptorch.Options().deviceIterations(1)
    poptorch_model = poptorch.trainingModel(
        model,
        opts,
        loss=torch.nn.CrossEntropyLoss(reduction="mean"),
        optimizer=optimizer)

    input = torch.randn(1, 10)
    label = torch.randint(0, 10, [1])

    # Make sure the first run doesn't already pass the test.
    original, original_loss = poptorch_model(input, label)

    # Loss shouldn't change.
    for i in range(0, 50):
        out, loss = poptorch_model(input, label)
        assert loss == original_loss

    # We shouldn't get the right result.
    assert not torch.argmax(out, dim=1) == label

    # Update the optimizer and check the loss now begins to decrease.
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    poptorch_model.setOptimizer(optimizer)
    poptorch_model(input, label)

    for i in range(0, 1000):
        out, loss = poptorch_model(input, label)

    # Check we have trained the "model"
    assert loss < original_loss
    assert loss < 0.03
    assert torch.argmax(out, dim=1) == label
