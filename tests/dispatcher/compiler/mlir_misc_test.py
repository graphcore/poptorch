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
