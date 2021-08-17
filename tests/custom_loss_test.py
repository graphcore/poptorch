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

    base_model = torch.nn.Linear(10, 10)

    class CustomLoss(torch.nn.Module):
        # Mean squared error scaled.
        def forward(self, x, target):
            partial_loss = poptorch.identity_loss(x - target,
                                                  reduction=reduction)
            loss = partial_loss * partial_loss * 5
            return partial_loss, poptorch.identity_loss(loss, reduction="mean")

    loss_fn = CustomLoss()

    class ModelWithLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = base_model

        def forward(self, data, target):
            out = base_model(data)
            loss = loss_fn(out, target)
            return out, loss

    model = ModelWithLoss()
    poptorch_model = poptorch.trainingModel(model)

    target = torch.randn(10)
    input = torch.randn(10)

    # Capture what the loss function will see before the loss changes.
    x, _ = model(input, target)
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


# Test custom loss by training to targets
def run_custom_loss_test(loss_fn,
                         base_model=torch.nn.Linear(10, 10),
                         input=torch.randn(1, 10),
                         target=torch.randint(0, 10, [1]),
                         test_output_vs_target=True):
    torch.manual_seed(42)

    class ModelWithLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = base_model
            self.loss_fn = loss_fn

        def forward(self, data, target):
            out = base_model(data)
            loss = self.loss_fn(out, target)
            return out, loss

    model = ModelWithLoss()
    poptorch_model = poptorch.trainingModel(model)

    # Pytorch native.
    native_out, loss = model(input, target)

    #Make sure the first run doesn't already pass the test.
    original, original_loss = poptorch_model(input, target)

    assert original_loss > 0.1

    if test_output_vs_target:
        assert not torch.allclose(native_out, target, rtol=1e-02, atol=1e-02)
        assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)

    for _ in range(0, 2500):
        out, loss = poptorch_model(input, target)

    # Check we have trained the "model"
    assert loss < 0.01

    if test_output_vs_target:
        helpers.assert_allclose(actual=out,
                                expected=target,
                                rtol=1e-02,
                                atol=1e-02)

        # Check that the pytorch native model is also returning the trained
        # value that was trained on IPU.
        out, _ = model(input, target)
        helpers.assert_allclose(actual=out,
                                expected=target,
                                rtol=1e-02,
                                atol=1e-02)

    return poptorch_model


def test_custom_loss():
    class CustomLoss(torch.nn.Module):
        # Mean squared error scaled.
        def forward(self, x, target):
            loss = poptorch.identity_loss(x - target, reduction="none")
            loss = loss * loss * 5.0
            return poptorch.identity_loss(loss, reduction="mean")

    run_custom_loss_test(loss_fn=CustomLoss(),
                         input=torch.randn(10),
                         target=torch.randn(10))


def test_custom_loss_l1():
    class CustomLoss(torch.nn.Module):
        # Mean squared error scaled.
        def forward(self, x, target):
            loss = torch.nn.functional.l1_loss(x, target)
            loss = loss * loss * 5.0
            return poptorch.identity_loss(loss, reduction="mean")

    run_custom_loss_test(loss_fn=CustomLoss(),
                         input=torch.randn(10),
                         target=torch.randn(10))


def test_custom_loss_nll():
    class CustomLoss(torch.nn.Module):
        # Mean squared error scaled.
        def forward(self, x, target):
            loss = torch.nn.functional.nll_loss(x, target)
            loss = loss * 5.0
            return poptorch.identity_loss(loss, reduction="mean")

    base_model = torch.nn.Sequential(torch.nn.Linear(10, 10),
                                     torch.nn.LogSoftmax(dim=1))

    input = torch.randn(1, 10)
    target = torch.randint(0, 10, [1])

    out = base_model(input)

    model = run_custom_loss_test(loss_fn=CustomLoss(),
                                 base_model=base_model,
                                 input=input,
                                 target=target,
                                 test_output_vs_target=False)
    model.copyWeightsToHost()

    # Check that the pytorch native model is also returning the trained
    # value that was trained on IPU.
    out = base_model(input)

    assert torch.argmax(out, dim=1) == target


def test_two_custom_losses():
    base_model = torch.nn.Sequential(torch.nn.Linear(10, 10),
                                     torch.nn.LogSoftmax(dim=1))

    class CustomLoss(torch.nn.Module):
        # Mean squared error scaled.
        def forward(self, x, target):
            loss = torch.nn.functional.nll_loss(x, target)
            loss2 = torch.nn.functional.nll_loss(x, target) * 5.0
            return loss + loss2

    error_msg = ("Multiple independent losses found in graph. "
                 "Graph must have one final loss. "
                 "Wrap final graph loss in poptorch.identity_loss.")
    with pytest.raises(poptorch.Error, match=error_msg):
        run_custom_loss_test(loss_fn=CustomLoss(), base_model=base_model)


def test_two_custom_losses_with_id_wrapper():
    base_model = torch.nn.Sequential(torch.nn.Linear(10, 10),
                                     torch.nn.LogSoftmax(dim=1))

    class CustomLoss(torch.nn.Module):
        # Mean squared error scaled.
        def forward(self, x, target):
            loss = torch.nn.functional.nll_loss(x, target)
            loss2 = torch.nn.functional.nll_loss(x, target) * 5.0
            return poptorch.identity_loss(loss + loss2, reduction="mean")

    run_custom_loss_test(loss_fn=CustomLoss(),
                         base_model=base_model,
                         test_output_vs_target=False)


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
