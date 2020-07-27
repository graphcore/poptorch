#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.optim as optim
import poptorch


# Test L1 loss by directly running it against the pytorch native L1 in inference.
def test_L1Loss_direct():
    reductions = ["mean", "sum"]
    torch.manual_seed(42)

    for reduction in reductions:
        model = torch.nn.L1Loss(reduction=reduction)

        poptorch_model = poptorch.inferenceModel(model)

        for _ in range(0, 10):
            target = torch.randn(10)
            input = torch.randn(10)

            groundTruth = model(target, input)
            poptorch_out = poptorch_model(target, input)

            assert torch.allclose(groundTruth, poptorch_out)


# Test L1 loss by using it to match a target label
def test_L1Loss_training():
    torch.manual_seed(42)

    reductions = ["mean", "sum"]

    for reduction in reductions:
        torch.manual_seed(42)

        model = torch.nn.Linear(10, 10)

        poptorch_model = poptorch.trainingModel(
            model,
            loss=torch.nn.L1Loss(reduction=reduction),
            optimizer=optim.SGD(model.parameters(), lr=0.01))

        target = torch.randn(10)
        input = torch.randn(10)

        # Make sure the first run doesn't already pass the test.
        original, original_loss = poptorch_model(input, target)
        assert original_loss > 0.1
        assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)

        for i in range(0, 2000):
            out, loss = poptorch_model(input, target)

            # Model needs to adjust the LR in the middle to converge
            if i == 1000:
                poptorch_model.setOptimizer(
                    optim.SGD(model.parameters(), lr=0.001))

        # Check we have trained the "model"
        assert loss < original_loss

        # "sum" L1 losses tend to be very large compared to "mean"
        if reduction == "sum":
            assert loss < 0.1
        else:
            assert loss < 0.001

        assert torch.allclose(out, target, rtol=1e-02, atol=1e-02)


# Test MSE loss by directly running it against the pytorch native MSE in inference.
def test_MSELoss_direct():

    reductions = ["mean", "sum"]
    torch.manual_seed(42)

    for reduction in reductions:
        model = torch.nn.MSELoss(reduction=reduction)

        poptorch_model = poptorch.inferenceModel(model)

        for _ in range(0, 10):
            target = torch.randn(10)
            input = torch.randn(10)

            groundTruth = model(target, input)
            poptorch_out = poptorch_model(target, input)

            assert torch.allclose(groundTruth, poptorch_out)


# Test MSE loss by using it to match a target label
def test_MSELoss_training():
    torch.manual_seed(42)

    model = torch.nn.Linear(10, 10)

    poptorch_model = poptorch.trainingModel(model, loss=torch.nn.MSELoss())

    target = torch.randn(10)
    input = torch.randn(10)

    # Make sure the first run doesn't already pass the test.s
    original, original_loss = poptorch_model(input, target)
    assert original_loss > 0.1
    assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)

    for _ in range(0, 2500):
        out, loss = poptorch_model(input, target)

    # Check we have trained the "model"
    assert loss < 0.001
    assert torch.allclose(out, target, rtol=1e-02, atol=1e-02)


# This also servees as the NLL loss test as it uses NLL under the hood.
def test_CrossEntropy_direct():
    reductions = ["mean", "sum"]
    torch.manual_seed(42)

    for reduction in reductions:
        model = torch.nn.CrossEntropyLoss(reduction=reduction)

        poptorch_model = poptorch.inferenceModel(model)

        for _ in range(0, 10):
            label = torch.randint(0, 10, [1])
            input = torch.randn(1, 10)

            groundTruth = model(input, label)
            poptorch_out = poptorch_model(input, label)
            assert torch.allclose(groundTruth, poptorch_out)


# Since we swap out the log to conform to popart we are going to need to check that the LogSoftmax still actually works in normal contexts AND can be fed into a loss at the same time.
def test_LogSoftmax():
    torch.manual_seed(42)

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.softmax = torch.nn.LogSoftmax(dim=1)

        def forward(self, x):
            x = self.linear(x)
            return self.softmax(x)

    model = Net()

    poptorch_model = poptorch.trainingModel(
        model, loss=torch.nn.NLLLoss(reduction="mean"))

    for _ in range(0, 10):
        label = torch.randint(0, 10, [1])
        input = torch.randn(1, 10)

        # Run on host.
        groundTruth = model(input)
        poptorch_out, _ = poptorch_model(input, label)

        assert torch.allclose(groundTruth, poptorch_out)


# Test NLL loss by using it to match a target label.
def test_NLLLoss_training():

    reductions = ["mean", "sum"]
    torch.manual_seed(42)

    for reduction in reductions:

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.softmax = torch.nn.LogSoftmax(dim=1)

            def forward(self, x):
                x = self.linear(x)
                return self.softmax(x)

        model = Net()

        poptorch_model = poptorch.trainingModel(
            model, loss=torch.nn.NLLLoss(reduction=reduction))
        input = torch.randn(1, 10)
        label = torch.randint(0, 10, [1])

        # Make sure the first run doesn't already pass the test.
        _, original_loss = poptorch_model(input, label)

        for _ in range(0, 1000):
            out, loss = poptorch_model(input, label)

        # # Check we have trained the "model"
        assert loss < original_loss
        assert torch.argmax(out, dim=1) == label


# Test CrossEntropyLoss loss by using it to match a target label.
def test_CrossEntropyLoss_training():
    torch.manual_seed(42)

    reductions = ["mean", "sum"]
    torch.manual_seed(42)

    for reduction in reductions:
        model = torch.nn.Linear(10, 10)

        poptorch_model = poptorch.trainingModel(
            model, loss=torch.nn.CrossEntropyLoss(reduction=reduction))
        input = torch.randn(1, 10)
        label = torch.randint(0, 10, [1])

        # Make sure the first run doesn't already pass the test.
        _, original_loss = poptorch_model(input, label)

        for _ in range(0, 1000):
            out, loss = poptorch_model(input, label)

        # # Check we have trained the "model"
        assert loss < original_loss
        assert torch.argmax(out, dim=1) == label


# This also servees as the NLL loss test as it uses NLL under the hood.
def test_BCE_direct():
    reductions = ["mean", "sum"]
    torch.manual_seed(42)

    for reduction in reductions:
        model = torch.nn.BCELoss(reduction=reduction)

        poptorch_model = poptorch.inferenceModel(model)

        for _ in range(0, 10):
            target = torch.empty(10).random_(2)
            input = torch.empty(10).uniform_()

            groundTruth = model(input, target)
            poptorch_out = poptorch_model(input, target)
            assert torch.allclose(groundTruth, poptorch_out)


# TODO(T22975)
# This also servees as the NLL loss test as it uses NLL under the hood.
# Re-enable once pytorch fixes https://github.com/pytorch/pytorch/issues/40679
# def test_BCE_direct_with_weight():
#     reductions = ["mean", "sum"]
#     torch.manual_seed(42)

#     for reduction in reductions:

#         weight = torch.randn(10)
#         model = torch.nn.BCELoss(weight=weight, reduction=reduction)

#         poptorch_model = poptorch.inferenceModel(model)

#         for i in range(0, 10):
#             target = torch.empty(10, 10).random_(2)
#             input = torch.empty(10, 10).uniform_()

#             groundTruth = model(input, target)
#             poptorch_out = poptorch_model(input, target)
#             assert torch.allclose(groundTruth, poptorch_out)


def test_BCE_training():
    torch.manual_seed(42)

    reductions = ["mean", "sum"]
    torch.manual_seed(42)

    for reduction in reductions:
        model = torch.nn.Sequential(torch.nn.Linear(10, 10),
                                    torch.nn.Sigmoid())

        poptorch_model = poptorch.trainingModel(
            model,
            loss=torch.nn.BCELoss(reduction=reduction),
            optimizer=optim.SGD(model.parameters(), lr=0.1))

        target = torch.empty(10).uniform_()
        input = torch.randn(10)

        # Make sure the first run doesn't already pass the test.
        _, original_loss = poptorch_model(input, target)

        for _ in range(0, 1000):
            out, loss = poptorch_model(input, target)

        print(out)
        print(target)
        print(loss)
        print("\n")

        # # Check we have trained the "model"
        assert loss < original_loss
        torch.testing.assert_allclose(target, out, rtol=1e-03, atol=1e-03)
