#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import helpers
import poptorch
from poptorch.experimental import IPUContext


@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("K", [1, 2, 3, 4])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_topk(largest, K, dim):
    torch.manual_seed(42)

    input = torch.randn([2, 3, 4, 10])

    ipu_value, ipu_indices = IPUContext(torch.topk)(input, K, dim, largest)
    cpu_value, cpu_indices = torch.topk(input, K, dim, largest)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=cpu_value,
                            actual=ipu_value,
                            equal_nan=True)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=cpu_indices,
                            actual=ipu_indices,
                            equal_nan=True)
