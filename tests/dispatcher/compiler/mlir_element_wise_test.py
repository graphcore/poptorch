#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import helpers
from poptorch.experimental import IPUContext


def gt(t1, t2):
    return t1 > t2


def lt(t1, t2):
    return t1 < t2


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("op", [gt, lt])
def test_gt_lt(op):
    torch.manual_seed(42)

    t1 = torch.randn(3, 3)
    t2 = torch.randn(3, 3)

    cpu_result = op(t1, t2)
    ipu_result = IPUContext(op)(t1, t2)

    helpers.assert_allequal(actual=ipu_result, expected=cpu_result)
