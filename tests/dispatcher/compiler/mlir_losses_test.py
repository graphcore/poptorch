#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
import torch.nn.functional as F
import pytest
import helpers
import poptorch


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("ignore_index", [2, -100])
def test_nll_loss_forward(reduction, ignore_index):
    torch.manual_seed(42)
    input1 = torch.randn([4, 10])
    input2 = torch.Tensor([1, 2, 3, 4]).long()
    with poptorch.IPUScope([input1, input2],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        out = F.nll_loss(input1,
                         input2,
                         reduction=reduction,
                         ignore_index=ignore_index)
        ipu.outputs([out])

    cpu_result = F.nll_loss(input1,
                            input2,
                            reduction=reduction,
                            ignore_index=ignore_index)
    ipu_result = ipu(input1, input2)
    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result,
                            actual=cpu_result,
                            equal_nan=True)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("ignore_index", [2, -100])
def test_nll_loss_backward(reduction, ignore_index):
    torch.manual_seed(42)
    input1 = torch.nn.parameter.Parameter(torch.randn([4, 10]))
    input2 = torch.Tensor([1, 2, 3, 4]).long()
    with poptorch.IPUScope([input1.data, input2],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        loss = F.nll_loss(input1,
                          input2,
                          reduction=reduction,
                          ignore_index=ignore_index)
        if reduction == "none":
            loss = loss.sum()
        loss.backward()
        ipu.outputs([input1.grad])

    input1.grad.zero_()
    input1.grad.detach_()
    loss = F.nll_loss(input1,
                      input2,
                      reduction=reduction,
                      ignore_index=ignore_index)
    if reduction == "none":
        loss = loss.sum()
    loss.backward()
    cpu_result = input1.grad

    ipu_result = ipu(input1, input2)
    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result,
                            actual=cpu_result,
                            equal_nan=True)
