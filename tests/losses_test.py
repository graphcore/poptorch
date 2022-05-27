#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os  # pylint: disable=unused-import
import unittest.mock
import torch
import torch.optim as optim
import torch.nn.functional as F
import pytest
import poptorch
import helpers


def loss_harness(trace_model,
                 loss,
                 inputs,
                 target,
                 reduction,
                 op=None,
                 training=True,
                 **kwargs):

    if len(inputs) == 1:
        loss_fn = lambda x: loss(x, target, reduction=reduction, **kwargs)

        if op is None:
            op = lambda x: x
    elif len(inputs) == 2:
        loss_fn = lambda x, y: loss(
            x, y, target, reduction=reduction, **kwargs)

        if op is None:
            op = lambda x, y: (x, y)

    else:
        assert len(inputs) == 3
        # The only supported loss fn with 3 inputs is TripletMarginLoss
        # which has no "target" per se
        loss_fn = lambda x, y, z: loss(x, y, z, reduction=reduction, **kwargs)

        if op is None:
            op = lambda x, y, z: (x, y, z)

    model = helpers.ModelWithWeights(op, inputs[0].shape, loss_fn=loss_fn)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(
        model) if training else poptorch.inferenceModel(model, trace_model)

    native_out, _ = model(tuple(inputs))
    poptorch_out, poptorch_loss = poptorch_model(tuple(inputs))

    # Inference test - check outputs
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    if training:
        # Training test - check weights have changed
        poptorch_model.assert_weights_changed()

    # Return the poptorch model and original outputs for any further
    # testing
    return poptorch_model, poptorch_out, poptorch_loss


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("trace_model", [True, False])
def test_L1Loss(reduction, trace_model):
    torch.manual_seed(42)

    target = torch.randn(10)
    input = torch.randn(10)

    poptorch_model, original, original_loss = loss_harness(
        trace_model, F.l1_loss, [input], target, reduction)

    # Make sure the first run doesn't already pass the test.
    assert original_loss > 0.1
    assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)

    for i in range(0, 1000):
        out, loss = poptorch_model(input)

        # Model needs to adjust the LR in the middle to converge
        if i == 500:
            poptorch_model.setOptimizer(
                optim.SGD(poptorch_model.model.parameters(), lr=0.001))

    # Check we have trained the "model"
    assert loss < original_loss

    # "sum" L1 losses tend to be very large compared to "mean"
    if reduction == "sum":
        assert loss < 0.1
    else:
        assert loss < 0.001

    helpers.assert_allclose(actual=out,
                            expected=target,
                            rtol=1e-02,
                            atol=1e-02)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("trace_model", [True, False])
def test_MSELoss(reduction, trace_model):
    torch.manual_seed(42)

    target = torch.randn(10)
    input = torch.randn(10)

    poptorch_model, original, original_loss = loss_harness(
        trace_model, F.mse_loss, [input], target, reduction)

    # Make sure the first run doesn't already pass the test
    assert original_loss > 0.1
    assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)

    for _ in range(0, 1000):
        out, loss = poptorch_model(input)

    # Check we have trained the "model"
    assert loss < 0.001
    helpers.assert_allclose(actual=out,
                            expected=target,
                            rtol=1e-02,
                            atol=1e-02)


cross_entropy_params = [
    # Input shape, reduction
    ((1, 10), "mean"),
    ((1, 10, 2), "sum"),
    ((1, 10, 2, 3), "mean"),
]


@pytest.mark.parametrize("input_shape, reduction", cross_entropy_params)
@pytest.mark.parametrize("trace_model", [True, False])
def test_CrossEntropy(input_shape, reduction, trace_model):
    torch.manual_seed(42)

    input = torch.randn(input_shape)
    label_shape = [input_shape[0]]
    if len(input_shape) > 2:
        label_shape.extend(input_shape[2:])
    label = torch.randint(0, 10, label_shape)

    poptorch_model, _, original_loss = loss_harness(trace_model,
                                                    F.cross_entropy, [input],
                                                    label, reduction)

    for _ in range(0, 100):
        out, loss = poptorch_model(input)

    # Check we have trained the "model"
    assert loss < original_loss
    helpers.assert_allequal(actual=torch.argmax(out, dim=1), expected=label)


# Test softmax and logsoftmax for dimensions more than 2
def op_withdim(trace_model, op, input):

    # Run on CPU.
    native_out = op(input)

    # Run on IPU.
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(op, options)
    poptorch_out = poptorch_model(input)

    helpers.assert_allclose(expected=native_out, actual=poptorch_out)


ops_float = [
    torch.nn.Softmax,
    torch.nn.LogSoftmax,
]


@unittest.mock.patch.dict("os.environ", helpers.disableSmallModel())
@pytest.mark.parametrize("op", ops_float)
@pytest.mark.parametrize("dim", range(-4, 3))
@pytest.mark.parametrize("trace_model", [True, False])
def test_op_withdim_4d(op, dim, trace_model):
    N, C = 11, 22
    M, K = 33, 44
    torch.manual_seed(42)
    x = torch.randn(N, C, M, K)

    op_withdim(trace_model, op(dim=dim), x)


@pytest.mark.parametrize("op", ops_float)
@pytest.mark.parametrize("dim", range(-2, 1))
@pytest.mark.parametrize("trace_model", [True, False])
def test_op_withdim_2d(op, dim, trace_model):
    N, C = 17, 13
    torch.manual_seed(42)
    x = torch.randn(N, C)

    op_withdim(trace_model, op(dim=dim), x)


# Test NLL loss by using it to match a target label.
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("trace_model", [True, False])
def test_NLLLoss(reduction, trace_model):
    torch.manual_seed(42)

    op = lambda x: F.log_softmax(x, dim=1)

    label = torch.randint(0, 10, [1])
    input = torch.randn(1, 10)

    poptorch_model, _, original_loss = loss_harness(trace_model, F.nll_loss,
                                                    [input], label, reduction,
                                                    op)

    for _ in range(0, 100):
        out, loss = poptorch_model(input)

    # Check we have trained the "model"
    assert loss < original_loss
    assert torch.argmax(out, dim=1) == label


# Test NLL loss 2d by using it to match a target label.
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("trace_model", [True, False])
def test_NLLLoss2d(reduction, trace_model):

    torch.manual_seed(42)
    N, C, M = 3, 2, 5

    op = lambda x: F.log_softmax(x, dim=1)

    y = torch.empty(N, M, M, dtype=torch.long).random_(0, C)
    x = torch.randn(N, C, M, M)

    poptorch_model, _, original_loss = loss_harness(trace_model, F.nll_loss,
                                                    [x], y, reduction, op)

    for _ in range(0, 100):
        out, loss = poptorch_model(x)

    # Check we have trained the "model"
    assert loss < original_loss
    helpers.assert_allclose(actual=torch.argmax(out, dim=1), expected=y)


# This also servees as the NLL loss test as it uses NLL under the hood.
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("trace_model", [True, False])
def test_BCE(reduction, trace_model):
    torch.manual_seed(42)

    target = torch.empty(10).uniform_()
    input = torch.randn(10)

    poptorch_model, _, original_loss = loss_harness(trace_model,
                                                    F.binary_cross_entropy,
                                                    [input],
                                                    target,
                                                    reduction,
                                                    op=torch.sigmoid)

    # Make sure the first run doesn't already pass the test.
    _, original_loss = poptorch_model(input)

    for _ in range(0, 2500):
        out, loss = poptorch_model(input)

    # # Check we have trained the "model"
    assert loss < original_loss
    helpers.assert_allclose(actual=out,
                            expected=target,
                            rtol=1e-02,
                            atol=1e-02)


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
#             helpers.assert_allclose(expected=groundTruth, actual=poptorch_out)


@pytest.mark.parametrize("reduction", {"mean", "sum", "batchmean"})
@pytest.mark.parametrize("log_target", {True, False})
@pytest.mark.parametrize("trace_model", [True, False])
def test_KLDiv(reduction, log_target, trace_model):
    torch.manual_seed(42)

    # 2D Tensors to test batchmean
    target = torch.empty(3, 10).uniform_()
    input = torch.randn(3, 10)

    loss_harness(trace_model,
                 F.kl_div, [input],
                 target,
                 reduction,
                 log_target=log_target)


@pytest.mark.parametrize("reduction", {"mean", "sum"})
@pytest.mark.parametrize("log_input", {True, False})
@pytest.mark.parametrize("full", {True, False})
@pytest.mark.parametrize("trace_model", [True, False])
def test_PoissonNLLLoss(reduction, log_input, full, trace_model):
    torch.manual_seed(42)

    target = torch.poisson(torch.rand(10) * 5)
    input = torch.empty(10).uniform_()

    loss_harness(trace_model,
                 F.poisson_nll_loss, [input],
                 target,
                 reduction,
                 full=full,
                 log_input=log_input)


@pytest.mark.parametrize("reduction", {"mean", "sum"})
@pytest.mark.parametrize("trace_model", [True, False])
def test_HingeEmbeddingLoss(reduction, trace_model):
    torch.manual_seed(42)

    delta = torch.rand(1) + 0.5

    # Generate random set of 1s and -1s for labels
    target = torch.randint(2, [10]) * 2 - 1
    input = torch.empty(10).uniform_()

    loss_harness(trace_model,
                 F.hinge_embedding_loss, [input],
                 target,
                 reduction,
                 margin=delta.item())


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


@pytest.mark.parametrize("reduction", {"mean", "sum"})
@pytest.mark.parametrize("input, target, weight, pos_weight",
                         params_bcewithlogits)
@pytest.mark.parametrize("trace_model", [True, False])
def test_BCEWithLogitsLoss(reduction, input, target, weight, pos_weight,
                           trace_model):

    loss_harness(trace_model,
                 F.binary_cross_entropy_with_logits, [input],
                 target,
                 reduction,
                 weight=weight,
                 pos_weight=pos_weight)


@pytest.mark.parametrize("reduction", {"mean", "sum"})
@pytest.mark.parametrize("trace_model", [True, False])
def test_SmoothL1Loss(reduction, trace_model):
    torch.manual_seed(42)

    input = torch.randn(10)
    target = torch.empty(10).uniform_()

    loss_harness(trace_model, F.smooth_l1_loss, [input], target, reduction)


@pytest.mark.parametrize("reduction", {"mean", "sum"})
@pytest.mark.parametrize("trace_model", [True, False])
def test_SoftMarginLoss(reduction, trace_model):
    torch.manual_seed(42)

    input = torch.empty(10).uniform_()
    # Generate random set of 1s and -1s for labels
    target = torch.randint(2, [10]) * 2 - 1

    loss_harness(trace_model, F.soft_margin_loss, [input], target, reduction)


# TODO(T30688): Support MultiLabelSoftMarginLoss
@pytest.mark.skip()
@pytest.mark.parametrize("reduction", {"mean", "sum"})
@pytest.mark.parametrize("specify_weight", {True, False})
@pytest.mark.parametrize("trace_model", [True, False])
def test_MultiLabelSoftMarginLoss(reduction, specify_weight, trace_model):
    torch.manual_seed(42)

    weight = torch.randn(3, 10) if specify_weight else None

    input = torch.empty(3, 10).uniform_()
    # Generate random set of 0s and 1s for labels
    target = torch.randint(2, [3, 10])

    loss_harness(trace_model,
                 F.multilabel_soft_margin_loss, [input],
                 target,
                 reduction,
                 weight=weight)


@pytest.mark.parametrize("reduction", {"mean", "sum"})
@pytest.mark.parametrize("trace_model", [True, False])
def test_CosineEmbeddingLoss(reduction, trace_model):
    torch.manual_seed(42)

    # Margin should be between -1 and 1
    margin = torch.rand(1) * 2 - 1

    input1 = torch.empty(10, 3).uniform_()
    input2 = torch.empty(10, 3).uniform_()

    # Generate random set of 1s and -1s for labels
    target = torch.randint(2, [10]) * 2 - 1

    loss_harness(trace_model,
                 F.cosine_embedding_loss, [input1, input2],
                 target,
                 reduction,
                 margin=margin.item())


@pytest.mark.parametrize("reduction", {"mean", "sum"})
@pytest.mark.parametrize("trace_model", [True, False])
def test_MarginRankingLoss(reduction, trace_model):
    torch.manual_seed(42)

    # Margin should be between -1 and 1
    margin = torch.rand(1) * 2 - 1

    # As per the current PyTorch implementation, both dims must be equal
    input1 = torch.empty(10, 10).uniform_()
    input2 = torch.empty(10, 10).uniform_()

    # Generate random set of 1s and -1s for labels
    target = torch.randint(2, [10]) * 2 - 1

    loss_harness(trace_model,
                 F.margin_ranking_loss, [input1, input2],
                 target,
                 reduction,
                 margin=margin.item())


@pytest.mark.parametrize("p", {2., 3.})
@pytest.mark.parametrize("swap", {True, False})
@pytest.mark.parametrize("reduction", {"mean", "sum"})
@pytest.mark.parametrize("trace_model", [True, False])
def test_TripletMarginLoss(p, swap, reduction, trace_model):
    torch.manual_seed(42)

    # Between 0 and 2
    margin = torch.rand(1) * 2

    anchor = torch.randn(10, 5)
    positive = torch.randn(10, 5)
    negative = torch.randn(10, 5)

    loss_harness(trace_model,
                 F.triplet_margin_loss, [anchor, positive, negative],
                 None,
                 reduction,
                 margin=margin.item(),
                 p=p,
                 swap=swap)


@pytest.mark.parametrize("blank", {0, 3})
@pytest.mark.parametrize("reduction", {"mean", "sum"})
@pytest.mark.parametrize("trace_model", [True, False])
@pytest.mark.parametrize("zero_infinity", [True, False])
def test_CTCLoss(blank, reduction, trace_model, zero_infinity):

    T = 10  # Input sequence length
    N = 4  # Batch size
    C = 5  # Number of classes
    S = 6 if not zero_infinity else 10  # Target sequence length
    S_min = 3  # Minimum target length

    torch.manual_seed(42)

    # Initialize random batch of input vectors, for *size = (T,N,C)
    input = torch.randn(T, N, C).log_softmax(2).detach()
    input_lengths = torch.full(size=(N, ), fill_value=T, dtype=torch.long)

    target_lengths = torch.randint(low=S_min,
                                   high=S,
                                   size=(N, ),
                                   dtype=torch.long)

    # Initialize random batch of targets (0..C excluding the blank class)
    target = torch.randint(low=0, high=C - 1, size=(N, S), dtype=torch.long)
    target[target > blank] += 1

    loss_harness(trace_model,
                 F.ctc_loss, [input],
                 target,
                 reduction,
                 input_lengths=input_lengths,
                 target_lengths=target_lengths,
                 blank=blank,
                 zero_infinity=zero_infinity)


@pytest.mark.parametrize("reduction", ("mean", "sum"))
def test_identity_with_linear_out_returned(reduction):
    torch.manual_seed(42)

    el_in = 2

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(el_in, el_in)

        def forward(self, x):
            out = self.lin(x)
            loss = poptorch.identity_loss(out, reduction=reduction)
            return loss, out

    x = torch.rand(1, 1, el_in)

    model = Model()
    native_loss, native_out = model(x)

    poptorch_model = poptorch.trainingModel(model)
    poptorch_loss, poptorch_out = poptorch_model(x)

    helpers.assert_allclose(actual=poptorch_loss, expected=native_loss)
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    assert native_loss.shape != native_out.shape
