#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest

import torch

from poptorch.experimental import IPUContext

shape = (3, 4, 5)

select_param_sets = []
for d in range(-len(shape), len(shape)):
    for i in range(-shape[d], shape[d]):
        select_param_sets.append((d, i))


# torch.select.int
@pytest.mark.parametrize("dim_idx", select_param_sets)
@pytest.mark.mlirSupportRequired
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


# Test that the result returned from `select.int` is a view, not a copy of the
# input tensor.
@pytest.mark.mlirSupportRequired
def test_select_is_view():
    num_elems = 1
    for d in shape:
        num_elems *= d

    t_cpu = torch.linspace(1, num_elems, num_elems).view(shape)
    t_ipu = torch.empty(shape).copy_(t_cpu)

    def fn(t, d, i):
        res = t.select(d, i)
        t.add_(8.0)
        return res

    d = min(1, len(shape) - 1)
    i = min(1, shape[d] - 1)

    assert torch.equal(IPUContext(fn)(t_ipu, d, i), fn(t_cpu, d, i))


# torch.unbind
@pytest.mark.parametrize("dim", range(-len(shape), len(shape)))
@pytest.mark.mlirSupportRequired
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
