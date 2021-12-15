#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import helpers
import poptorch


@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("K", [1, 2, 3, 4])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_topk(largest, K, dim):
    torch.manual_seed(42)

    input = torch.randn([2, 3, 4, 10])

    with poptorch.IPUScope([input],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        values, indices = torch.topk(input, k=K, largest=largest, dim=dim)
        ipu.outputs([values, indices])

    ipu_value, ipu_indices = ipu(input)
    cpu_value, cpu_indices = torch.topk(input, k=K, largest=largest, dim=dim)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_value,
                            actual=cpu_value,
                            equal_nan=True)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_indices,
                            actual=cpu_indices,
                            equal_nan=True)
