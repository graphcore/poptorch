# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from itertools import product

import pytest
import torch
from torch_geometric.nn import Linear

from dense_utils import dense_harness

weight_inits = ['glorot', "uniform", 'kaiming_uniform', None]
bias_inits = ['zeros', None]


@pytest.mark.parametrize('weight,bias', product(weight_inits, bias_inits))
def test_linear(weight, bias):
    lin = Linear(16, 32, weight_initializer=weight, bias_initializer=bias)
    x = torch.randn(1, 4, 16)

    dense_harness(lin, x)
