#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest

import torch

import poptorch
from poptorch.experimental import IPUContext

shape = (3, 4, 5)

select_param_sets = []
for d in range(-len(shape), len(shape)):
    for i in range(-shape[d], shape[d]):
        select_param_sets.append((d, i))


# torch.select.int
@pytest.mark.parametrize("dim_idx", select_param_sets)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_select(dim_idx):
    num_elems = 1
    for d in shape:
        num_elems *= d

    t = torch.linspace(1, num_elems, num_elems).view(shape)

    def fn(t, d, i):
        return t.select(d, i)

    d = dim_idx[0]
    i = dim_idx[1]

    assert torch.equal(IPUContext(fn)(t, d, i), fn(t, d, i))


# torch.unbind
@pytest.mark.parametrize("dim", range(-len(shape), len(shape)))
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_unbind(dim):
    num_elems = 1
    for d in shape:
        num_elems *= d

    t = torch.linspace(1, num_elems, num_elems).view(shape)

    def fn(t, d):
        return t.unbind(d)

    cpu_res = fn(t, dim)
    ipu_res = IPUContext(fn)(t, dim)

    assert len(cpu_res) == len(ipu_res)

    for c, i in zip(cpu_res, ipu_res):
        assert torch.equal(i, c)
