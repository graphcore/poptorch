#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import pytest
import helpers
import poptorch


#  Test the reductions work as expected
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_non_final_loss_reductions(reduction):
    torch.manual_seed(42)

    model = torch.nn.Linear(10, 10)

    class CustomLoss(torch.nn.Module):
        # Mean squared error scaled.
        def forward(self, x, target):
            partial_loss = poptorch.identity_loss(x - target,
                                                  reduction=reduction)
            loss = partial_loss * partial_loss * 5
            return partial_loss, poptorch.identity_loss(loss, reduction="mean")

    loss = CustomLoss()

    poptorch_model = helpers.trainingModelWithLoss(model, loss=loss)

    target = torch.randn(10)
    input = torch.randn(10)

    # Capture what the loss function will see before the loss changes.
    x = model(input)
    _, (partial_loss, _) = poptorch_model(input, target)

    # Check we have actually reduced the loss
    if reduction != "none":
        assert torch.numel(partial_loss) == 1

    if reduction == "mean":
        simulated_loss = torch.mean(x - target)
    elif reduction == "sum":
        simulated_loss = torch.sum(x - target)
    elif reduction == "none":
        simulated_loss = x - target

    torch.testing.assert_allclose(simulated_loss.reshape_as(partial_loss),
                                  partial_loss,
                                  rtol=1e-02,
                                  atol=1e-02)


# Test custom loss by training to a target
def test_custom_loss():
    torch.manual_seed(42)

    model = torch.nn.Linear(10, 10)

    class CustomLoss(torch.nn.Module):
        # Mean squared error scaled.
        def forward(self, x, target):
            loss = poptorch.identity_loss(x - target, reduction="none")
            loss = loss * loss * 5.0
            return poptorch.identity_loss(loss, reduction="mean")

    loss = CustomLoss()

    poptorch_model = helpers.trainingModelWithLoss(model, loss=loss)

    target = torch.randn(10)
    input = torch.randn(10)

    # Make sure the first run doesn't already pass the test.
    original, original_loss = poptorch_model(input, target)

    assert original_loss > 0.1
    assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)
    # Pytorch native.
    out = model(input)
    assert not torch.allclose(out, target, rtol=1e-02, atol=1e-02)

    for _ in range(0, 2500):
        out, loss = poptorch_model(input, target)

    # Check we have trained the "model"
    assert loss < 0.001
    assert torch.allclose(out, target, rtol=1e-02, atol=1e-02)

    # Check that the pytorch native model is also returning the trained
    # value that was trained on IPU.
    out = model(input)
    assert torch.allclose(out, target, rtol=1e-02, atol=1e-02)


# Test custom loss by training to a target
def test_custom_loss_l1():
    torch.manual_seed(42)

    model = torch.nn.Linear(10, 10)

    class CustomLoss(torch.nn.Module):
        # Mean squared error scaled.
        def forward(self, x, target):
            loss = torch.nn.functional.l1_loss(x, target)
            loss = loss * loss * 5.0
            return poptorch.identity_loss(loss, reduction="mean")

    loss = CustomLoss()

    poptorch_model = helpers.trainingModelWithLoss(model, loss=loss)

    target = torch.randn(10)
    input = torch.randn(10)

    # Make sure the first run doesn't already pass the test.
    original, original_loss = poptorch_model(input, target)

    assert original_loss > 0.1
    assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)

    # Pytorch native.
    out = model(input)
    assert not torch.allclose(out, target, rtol=1e-02, atol=1e-02)

    for _ in range(0, 2500):
        out, loss = poptorch_model(input, target)

    # Check we have trained the "model"
    assert loss < 0.001
    torch.testing.assert_allclose(out, target, rtol=1e-02, atol=1e-02)

    # Check that the pytorch native model is also returning the trained
    # value that was trained on IPU.
    out = model(input)
    torch.testing.assert_allclose(out, target, rtol=1e-02, atol=1e-02)


# Test custom loss by training to a label
def test_custom_loss_nll():
    torch.manual_seed(42)

    model = torch.nn.Sequential(torch.nn.Linear(10, 10),
                                torch.nn.LogSoftmax(dim=1))

    class CustomLoss(torch.nn.Module):
        # Mean squared error scaled.
        def forward(self, x, target):
            loss = torch.nn.functional.nll_loss(x, target)
            loss = loss * 5.0
            return poptorch.identity_loss(loss, reduction="mean")

    loss = CustomLoss()

    poptorch_model = helpers.trainingModelWithLoss(model, loss=loss)

    label = torch.randint(0, 10, [1])
    input = torch.randn(1, 10)

    # Make sure the first run doesn't already pass the test.
    _, original_loss = poptorch_model(input, label)

    assert original_loss > 0.1

    # Pytorch native.
    out = model(input)

    for _ in range(0, 2500):
        out, loss = poptorch_model(input, label)

    # Check we have trained the "model"
    assert loss < 0.01
    # Check that the pytorch native model is also returning the trained
    # value that was trained on IPU.
    out = model(input)
    assert torch.argmax(out, dim=1) == label
