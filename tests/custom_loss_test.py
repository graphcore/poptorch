#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.optim as optim
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

    helpers.assert_allclose(expected=simulated_loss.reshape_as(partial_loss),
                            actual=partial_loss,
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
    helpers.assert_allclose(actual=out,
                            expected=target,
                            rtol=1e-02,
                            atol=1e-02)

    # Check that the pytorch native model is also returning the trained
    # value that was trained on IPU.
    out = model(input)
    helpers.assert_allclose(actual=out,
                            expected=target,
                            rtol=1e-02,
                            atol=1e-02)


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
    helpers.assert_allclose(actual=out,
                            expected=target,
                            rtol=1e-02,
                            atol=1e-02)

    # Check that the pytorch native model is also returning the trained
    # value that was trained on IPU.
    out = model(input)
    helpers.assert_allclose(actual=out,
                            expected=target,
                            rtol=1e-02,
                            atol=1e-02)


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


# Test custom loss by training to a label
def test_two_custom_losses():
    torch.manual_seed(42)

    model = torch.nn.Sequential(torch.nn.Linear(10, 10),
                                torch.nn.LogSoftmax(dim=1))

    class CustomLoss(torch.nn.Module):
        # Mean squared error scaled.
        def forward(self, x, target):
            loss = torch.nn.functional.nll_loss(x, target)
            loss2 = torch.nn.functional.nll_loss(x, target) * 5.0
            a = loss + loss2
            return a, loss

    loss = CustomLoss()

    poptorch_model = helpers.trainingModelWithLoss(model, loss=loss)

    label = torch.randint(0, 10, [1])
    input = torch.randn(1, 10)

    error_msg = ("Multiple independent losses found in graph."
                 " Graph must have one final loss. "
                 "Wrap final graph loss in poptorch.identity_loss.")
    with pytest.raises(poptorch.Error, match=error_msg):
        _ = poptorch_model(input, label)


def test_two_custom_losses_with_id_wrapper():
    torch.manual_seed(42)

    model = torch.nn.Sequential(torch.nn.Linear(10, 10),
                                torch.nn.LogSoftmax(dim=1))

    class CustomLoss(torch.nn.Module):
        # Mean squared error scaled.
        def forward(self, x, target):
            loss = torch.nn.functional.nll_loss(x, target)
            loss2 = torch.nn.functional.nll_loss(x, target) * 5.0
            a = poptorch.identity_loss(loss + loss2, reduction="mean")
            return a, loss

    loss = CustomLoss()

    poptorch_model = helpers.trainingModelWithLoss(model, loss=loss)

    label = torch.randint(0, 10, [1])
    input = torch.randn(1, 10)

    _ = poptorch_model(input, label)


def test_no_loss():
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(torch.nn.Linear(10, 10),
                                             torch.nn.LogSoftmax(dim=1))

        # Mean squared error scaled.
        def forward(self, x, target):
            fwd = self.model(x)
            loss = fwd * 12
            loss2 = target + 1
            a = loss + loss2
            return fwd, a, loss

    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    poptorch_model = poptorch.trainingModel(model, optimizer=optimizer)

    label = torch.randint(0, 10, [1])
    input = torch.randn(1, 10)

    error_msg = "Couldn't find a loss in graph"
    with pytest.raises(poptorch.Error, match=error_msg):
        _ = poptorch_model(input, label)
