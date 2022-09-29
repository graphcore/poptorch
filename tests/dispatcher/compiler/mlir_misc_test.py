#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import torch
import pytest
import poptorch
from poptorch.experimental import IPUContext


@pytest.mark.mlirSupportRequired
def test_ignored_values():
    def to_dtype(x: torch.Tensor):
        return torch.normal(x, x, generator=torch.Generator())

    msg = ('normal.Tensor_Tensor: Poptorch does not handle generator. '
           'Expected it to be None')
    with pytest.raises(poptorch.Error, match=msg):
        IPUContext(to_dtype)(torch.tensor(1.0))


@pytest.mark.mlirSupportRequired
def test_function_reuse():
    def f(x):
        return x + 1

    compiled = IPUContext(f)

    in1 = torch.tensor(1)
    out1 = compiled(in1)

    in2 = torch.tensor(2)
    out2 = compiled(in2)

    assert out1.item() == 2
    assert out2.item() == 3
