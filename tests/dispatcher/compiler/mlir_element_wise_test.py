#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import helpers
import poptorch


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_gt_lt():
    torch.manual_seed(42)

    t1 = torch.randn(3, 3)
    t2 = torch.randn(3, 3)

    cpu_gt = t1 > t2
    cpu_lt = t1 < t2

    with poptorch.IPUScope([t1, t2],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        ipu_gt = t1 > t2
        ipu_lt = t1 < t2
        ipu.outputs([ipu_gt, ipu_lt])

    ipu_gt, ipu_lt = ipu(t1, t2)

    # pylint: disable=no-member
    helpers.assert_allequal(actual=ipu_gt, expected=cpu_gt)
    # pylint: disable=no-member
    helpers.assert_allequal(actual=ipu_lt, expected=cpu_lt)
