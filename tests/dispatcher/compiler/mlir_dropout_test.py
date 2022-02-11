#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import math
import torch
import torch.nn.functional as F
import pytest
import poptorch
from poptorch.experimental import IPUContext


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_dropout_eval():
    # Dropout in eval mode should be an Identity operation
    torch.manual_seed(42)

    t1 = torch.randn(10)

    @IPUContext
    def ipu_dropout(t):
        return F.dropout(t, p=0.5, training=False)

    t2 = ipu_dropout(t1)
    assert (t1 == t2).all()


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="IPU Model's random generator is insufficient.")
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 1.0])
def test_dropout_train(p):
    torch.manual_seed(42)

    n = 1000
    t1 = torch.randn(n)

    @IPUContext
    def ipu_dropout(t):
        return F.dropout(t, p=p, training=True)

    t2 = ipu_dropout(t1)
    num_zeros = n - torch.count_nonzero(t2).item()

    if p > 0.0:
        assert (t1 != t2).all()
    else:
        # No zeros if p=0
        assert num_zeros == 0

    if p < 1.0:
        # Number of zeros should be within mean +/- 2*std of Binomial distribution
        sigma = math.sqrt(n * p * (1. - p))
        assert num_zeros >= (p * n - 2 * sigma)
        assert num_zeros <= (p * n + 2 * sigma)

        # Unmasked values should be scaled by 1/(1-p)
        nonzero_mask = t2 != 0.0
        assert (t2[nonzero_mask] == 1. / (1. - p) * t1[nonzero_mask]).all()
    else:
        # All zeros if p=1
        assert num_zeros == n
