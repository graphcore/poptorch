#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import helpers
from poptorch.experimental import IPUContext


@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("K", [1, 2, 3, 4])
@pytest.mark.parametrize("dim", [-1, -2])
def test_topk(largest, K, dim):
    torch.manual_seed(42)

    input = torch.randn([2, 3, 4, 10])

    ipu_value, ipu_indices = IPUContext(torch.topk)(input, K, dim, largest)
    cpu_value, cpu_indices = torch.topk(input, K, dim, largest)

    helpers.assert_allclose(expected=cpu_value,
                            actual=ipu_value,
                            equal_nan=True)

    helpers.assert_allclose(expected=cpu_indices,
                            actual=ipu_indices,
                            equal_nan=True)
