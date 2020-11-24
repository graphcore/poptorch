#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os  # pylint: disable=unused-import
import unittest.mock
import torch
import torch.optim as optim
import pytest
import poptorch
import helpers


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

        poptorch_model = helpers.trainingModelWithLoss(
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

    poptorch_model = helpers.trainingModelWithLoss(model,
                                                   loss=torch.nn.MSELoss())

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

    poptorch_model = helpers.trainingModelWithLoss(
        model, loss=torch.nn.NLLLoss(reduction="mean"))

    for _ in range(0, 10):
        label = torch.randint(0, 10, [1])
        input = torch.randn(1, 10)

        # Run on host.
        groundTruth = model(input)
        poptorch_out, _ = poptorch_model(input, label)

        assert torch.allclose(groundTruth, poptorch_out)


# Test softmax and logsoftmax for dimensions more than 2
def op_withdim(op, input):
    class Model(torch.nn.Module):
        def __init__(self, op):
            super(Model, self).__init__()
            self.op = op

        def forward(self, x):
            return self.op(x)

    model = Model(op)

    # Run on CPU.
    nativeOut = model(input)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input)

    torch.testing.assert_allclose(nativeOut, poptorch_out)


ops_float = [
    torch.nn.Softmax,
    torch.nn.LogSoftmax,
]


@unittest.mock.patch.dict("os.environ", helpers.disableSmallModel())
@pytest.mark.parametrize("op", ops_float)
@pytest.mark.parametrize("dim", range(-4, 3))
def test_op_withdim_4d(op, dim):
    N, C = 11, 22
    M, K = 33, 44
    torch.manual_seed(42)
    x = torch.randn(N, C, M, K)

    op_withdim(op(dim=dim), x)


@pytest.mark.parametrize("op", ops_float)
@pytest.mark.parametrize("dim", range(-2, 1))
def test_op_withdim_2d(op, dim):
    N, C = 17, 13
    torch.manual_seed(42)
    x = torch.randn(N, C)

    op_withdim(op(dim=dim), x)


# Test NLL loss by using it to match a target label.
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_NLLLoss_training(reduction):

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

    poptorch_model = helpers.trainingModelWithLoss(
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


# Test NLL loss 2d by using it to match a target label.
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_NLLLoss2d_training(reduction):

    torch.manual_seed(42)
    N, C, M = 3, 2, 5

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = torch.nn.Linear(M * M, M * M * C)
            self.softmax = torch.nn.LogSoftmax(dim=1)

        def forward(self, x):
            x = x.reshape(N, M * M)
            x = self.linear(x).reshape(N, C, M, M)
            return self.softmax(x)

    model = Net()

    poptorch_model = helpers.trainingModelWithLoss(
        model, loss=torch.nn.NLLLoss(reduction=reduction))
    x = torch.randn(N, M, M)
    y = torch.empty(N, M, M, dtype=torch.long).random_(0, C)

    _, original_loss = poptorch_model(x, y)

    for _ in range(0, 1000):
        out, loss = poptorch_model(x, y)

    # # Check we have trained the "model"
    assert loss < original_loss
    torch.testing.assert_allclose(torch.argmax(out, dim=1), y)


# Tell loss 2d in an inference model, comparing against pytorch
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_NLLLoss2d_inference(reduction):
    N, C, M = 11, 7, 13
    torch.manual_seed(42)

    class Net(torch.nn.Module):
        def __init__(self, reduction):
            super(Net, self).__init__()
            self.softmax = torch.nn.LogSoftmax(dim=1)
            self.loss = torch.nn.NLLLoss(reduction=reduction)

        def forward(self, x, y):
            x = self.softmax(x)
            return self.loss(x, y)

    model = Net(reduction)
    poptorch_model = poptorch.inferenceModel(model)

    x = torch.randn(N, C, M, M)
    y = torch.empty(N, M, M, dtype=torch.long).random_(0, C)

    native_out = model(x, y)
    poptorch_out = poptorch_model(x, y)

    torch.testing.assert_allclose(poptorch_out, native_out)


# Test CrossEntropyLoss loss by using it to match a target label.
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_CrossEntropyLoss_training(reduction):
    torch.manual_seed(42)

    torch.manual_seed(42)

    model = torch.nn.Linear(10, 10)

    poptorch_model = helpers.trainingModelWithLoss(
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
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_BCE_direct(reduction):
    torch.manual_seed(42)

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


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_BCE_training(reduction):
    torch.manual_seed(42)

    torch.manual_seed(42)

    model = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.Sigmoid())

    poptorch_model = helpers.trainingModelWithLoss(
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


@pytest.mark.parametrize("reduction", {"none", "mean", "sum", "batchmean"})
@pytest.mark.parametrize("log_target", {True, False})
def test_KLDiv_direct(reduction, log_target):
    torch.manual_seed(42)

    model = torch.nn.KLDivLoss(reduction=reduction, log_target=log_target)
    poptorch_model = poptorch.inferenceModel(model)

    # 2D Tensors to test batchmean
    target = torch.empty(3, 10).uniform_()
    input = torch.randn(3, 10)

    native_out = model(input, target)
    poptorch_out = poptorch_model(input, target)

    torch.testing.assert_allclose(native_out, poptorch_out)


@pytest.mark.parametrize("reduction", {"none", "mean", "sum"})
@pytest.mark.parametrize("log_input", {True, False})
@pytest.mark.parametrize("full", {True, False})
def test_PoissonNLLLoss_direct(reduction, log_input, full):
    torch.manual_seed(42)

    model = torch.nn.PoissonNLLLoss(log_input, full, reduction=reduction)
    poptorch_model = poptorch.inferenceModel(model)

    target = torch.poisson(torch.rand(10) * 5)
    input = torch.empty(10).uniform_()

    native_out = model(input, target)
    poptorch_out = poptorch_model(input, target)

    torch.testing.assert_allclose(native_out, poptorch_out)


@pytest.mark.parametrize("reduction", {"none", "mean", "sum"})
def test_HingeEmbeddingLoss_direct(reduction):
    torch.manual_seed(42)

    delta = torch.rand(1) + 0.5
    model = torch.nn.HingeEmbeddingLoss(delta.item(), reduction=reduction)
    poptorch_model = poptorch.inferenceModel(model)

    # Generate random set of 1s and -1s for labels
    target = torch.randint(2, [10]) * 2 - 1

    input = torch.empty(10).uniform_()

    native_out = model(input, target)
    poptorch_out = poptorch_model(input, target)

    torch.testing.assert_allclose(native_out, poptorch_out)


torch.manual_seed(42)
params_bcewithlogits = [
    (
        torch.rand(10, 3),  # Inputs
        torch.empty(10, 3).uniform_(),  # Targets
        torch.rand(10, 3),  # Weights
        torch.rand(3)  # Pos Weights
    ),
    # Numerical stability test
    (torch.tensor([88.0]), torch.tensor([0.5]), None, None)
]


@pytest.mark.parametrize("reduction", {"none", "mean", "sum"})
@pytest.mark.parametrize("params", params_bcewithlogits)
def test_BCEWithLogitsLoss_direct(reduction, params):

    weight = params[2]
    pos_weight = params[3]

    model = torch.nn.BCEWithLogitsLoss(weight=weight,
                                       reduction=reduction,
                                       pos_weight=pos_weight)
    poptorch_model = poptorch.inferenceModel(model)

    target = params[1]
    input = params[0]

    native_out = model(input, target)
    poptorch_out = poptorch_model(input, target)

    torch.testing.assert_allclose(native_out, poptorch_out)


@pytest.mark.parametrize("reduction", {"none", "mean", "sum"})
def test_SmoothL1Loss_direct(reduction):
    torch.manual_seed(42)

    model = torch.nn.SmoothL1Loss(reduction=reduction)
    poptorch_model = poptorch.inferenceModel(model)

    input = torch.randn(10)
    target = torch.empty(10).uniform_()

    native_out = model(input, target)
    poptorch_out = poptorch_model(input, target)

    torch.testing.assert_allclose(native_out, poptorch_out)


@pytest.mark.parametrize("reduction", {"none", "mean", "sum"})
def test_SoftMarginLoss_direct(reduction):
    torch.manual_seed(42)

    model = torch.nn.SoftMarginLoss(reduction=reduction)
    poptorch_model = poptorch.inferenceModel(model)

    input = torch.empty(10).uniform_()

    # Generate random set of 1s and -1s for labels
    target = torch.randint(2, [10]) * 2 - 1

    native_out = model(input, target)
    poptorch_out = poptorch_model(input, target)

    torch.testing.assert_allclose(native_out, poptorch_out)


# TODO(T30688): Support MultiLabelSoftMarginLoss
@pytest.mark.skip()
@pytest.mark.parametrize("reduction", {"none", "mean", "sum"})
@pytest.mark.parametrize("specify_weight", {True, False})
def test_MultiLabelSoftMarginLoss_direct(reduction, specify_weight):
    torch.manual_seed(42)

    weight = torch.randn(3, 10) if specify_weight else None

    model = torch.nn.MultiLabelSoftMarginLoss(weight, reduction=reduction)
    poptorch_model = poptorch.inferenceModel(model)

    input = torch.empty(3, 10).uniform_()

    # Generate random set of 0s and 1s for labels
    target = torch.randint(2, [3, 10])

    native_out = model(input, target)
    poptorch_out = poptorch_model(input, target)

    assert native_out.size() == poptorch_out.size()
    torch.testing.assert_allclose(native_out, poptorch_out)


@pytest.mark.parametrize("reduction", {"none", "mean", "sum"})
def test_CosineEmbeddingLoss_direct(reduction):
    torch.manual_seed(42)

    # Margin should be between -1 and 1
    margin = torch.rand(1) * 2 - 1

    model = torch.nn.CosineEmbeddingLoss(margin.item(), reduction=reduction)
    poptorch_model = poptorch.inferenceModel(model)

    # As per the current PyTorch implementation, the first two dims must be equal
    input1 = torch.empty(10, 10, 3).uniform_()
    input2 = torch.empty(10, 10, 3).uniform_()

    # Generate random set of 1s and -1s for labels
    target = torch.randint(2, [10, 10, 3]) * 2 - 1

    native_out = model(input1, input2, target)
    poptorch_out = poptorch_model(input1, input2, target)

    torch.testing.assert_allclose(native_out, poptorch_out)


@pytest.mark.parametrize("reduction", {"none", "mean", "sum"})
def test_MarginRankingLoss_direct(reduction):
    torch.manual_seed(42)

    # Margin should be between -1 and 1
    margin = torch.rand(1) * 2 - 1

    model = torch.nn.MarginRankingLoss(margin.item(), reduction=reduction)
    poptorch_model = poptorch.inferenceModel(model)

    # As per the current PyTorch implementation, both dims must be equal
    input1 = torch.empty(10, 10).uniform_()
    input2 = torch.empty(10, 10).uniform_()

    # Generate random set of 1s and -1s for labels
    target = torch.randint(2, [10]) * 2 - 1

    native_out = model(input1, input2, target)
    poptorch_out = poptorch_model(input1, input2, target)

    torch.testing.assert_allclose(native_out, poptorch_out)


@pytest.mark.parametrize("p", {2., 3.})
@pytest.mark.parametrize("swap", {True, False})
@pytest.mark.parametrize("reduction", {"none", "mean", "sum"})
def test_TripletMarginLoss_direct(p, swap, reduction):
    torch.manual_seed(42)

    # Between 0 and 2
    margin = torch.rand(1) * 2

    model = torch.nn.TripletMarginLoss(margin.item(),
                                       p,
                                       swap=swap,
                                       reduction=reduction)
    poptorch_model = poptorch.inferenceModel(model)

    anchor = torch.randn(10, 5)
    positive = torch.randn(10, 5)
    negative = torch.randn(10, 5)

    native_out = model(anchor, positive, negative)
    poptorch_out = poptorch_model(anchor, positive, negative)

    torch.testing.assert_allclose(native_out, poptorch_out)
