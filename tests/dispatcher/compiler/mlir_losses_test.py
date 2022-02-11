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
    # pylint: disable=no-member
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

    # pylint: disable=no-member
    helpers.assert_allclose(expected=cpu_result,
                            actual=ipu_result,
                            equal_nan=True)
