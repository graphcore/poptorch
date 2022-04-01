#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
import torch.nn.functional as F
import pytest
import helpers
import poptorch
from poptorch.experimental import IPUContext


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("ignore_index", [2, -100])
def test_nll_loss_forward(reduction, ignore_index):
    torch.manual_seed(42)
    input1 = torch.randn([4, 10])
    input2 = torch.Tensor([1, 2, 3, 4]).long()

    def nll_loss(t1, t2):
        return F.nll_loss(t1,
                          t2,
                          reduction=reduction,
                          ignore_index=ignore_index)

    cpu_result = nll_loss(input1, input2)
    ipu_result = IPUContext(nll_loss)(input1, input2)

    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            equal_nan=True)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("ignore_index", [2, -100])
def test_nll_loss_backward(reduction, ignore_index):
    torch.manual_seed(42)
    input1 = torch.nn.parameter.Parameter(torch.randn([4, 10]))
    input2 = torch.Tensor([1, 2, 3, 4]).long()

    def nll_loss_backward(t1, t2):
        loss = F.nll_loss(t1,
                          t2,
                          reduction=reduction,
                          ignore_index=ignore_index)
        if reduction == "none":
            loss = loss.sum()
        loss.backward()
        return t1.grad

    ipu_result = IPUContext(nll_loss_backward)(input1, input2)

    input1.grad.zero_()
    input1.grad.detach_()
    cpu_result = nll_loss_backward(input1, input2)

    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            equal_nan=True)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_binary_cross_entropy_forward(reduction):
    torch.manual_seed(42)
    input1 = torch.randn([4])
    input2 = torch.Tensor([1, 0, 0, 1])

    def bce(t1, t2):
        return F.binary_cross_entropy(t1, t2, reduction=reduction)

    ipu_result = IPUContext(bce)(input1, input2)
    cpu_result = bce(input1, input2)

    helpers.assert_allclose(expected=ipu_result,
                            actual=cpu_result,
                            equal_nan=True)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_binary_cross_entropy_backward(reduction):
    torch.manual_seed(42)
    input1 = torch.nn.parameter.Parameter(
        torch.clip(torch.randn([4]), min=0.001, max=0.999))
    input2 = torch.Tensor([1, 0, 0, 1])

    def bce_backward(t1, t2):
        loss = F.binary_cross_entropy(t1, t2, reduction=reduction)
        if reduction == "none":
            loss = loss.sum()
        loss.backward()
        return t1.grad

    ipu_result = IPUContext(bce_backward)(input1, input2)
    input1.grad.detach_()
    input1.grad.zero_()
    cpu_result = bce_backward(input1, input2)

    helpers.assert_allclose(expected=ipu_result,
                            actual=cpu_result,
                            equal_nan=True)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_binary_cross_entropy_with_logits_forward(reduction):
    torch.manual_seed(42)
    input1 = torch.randn([4])
    input2 = torch.Tensor([1, 0, 0, 1])

    def bce_logit(t1, t2):
        return F.binary_cross_entropy_with_logits(t1, t2, reduction=reduction)

    cpu_result = IPUContext(bce_logit)(input1, input2)
    ipu_result = (bce_logit)(input1, input2)

    helpers.assert_allclose(expected=ipu_result,
                            actual=cpu_result,
                            equal_nan=True)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_binary_cross_entropy_with_logits_backward(reduction):
    torch.manual_seed(42)
    input1 = torch.nn.parameter.Parameter(torch.randn([4]))
    input2 = torch.Tensor([1, 0, 0, 1])

    def bce_logit_backward(t1, t2):
        loss = F.binary_cross_entropy_with_logits(t1, t2, reduction=reduction)
        if reduction == "none":
            loss = loss.sum()
        loss.backward()
        return t1.grad

    ipu_result = IPUContext(bce_logit_backward)(input1, input2)
    input1.grad.detach_()
    input1.grad.zero_()
    cpu_result = bce_logit_backward(input1, input2)

    helpers.assert_allclose(expected=ipu_result,
                            actual=cpu_result,
                            equal_nan=True)
