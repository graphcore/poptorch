# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from itertools import product

import pytest
import torch
from torch_geometric.nn import HeteroLinear, Linear

from dense.dense_utils import dense_harness

weight_inits = ['glorot', "uniform", 'kaiming_uniform', None]
bias_inits = ['zeros', None]


@pytest.mark.parametrize('weight,bias', product(weight_inits, bias_inits))
def test_linear(weight, bias):
    lin = Linear(16, 32, weight_initializer=weight, bias_initializer=bias)
    x = torch.randn(1, 4, 16)

    dense_harness(lin, x)


@pytest.mark.parametrize('with_bias', [True, False])
def test_hetero_linear(with_bias):
    x = torch.randn(10, 16)
    type_vec = torch.tensor([0, 0, 2, 1, 0, 2, 2, 2, 1, 2])

    lin = HeteroLinear(16, 32, num_types=3, bias=with_bias)

    dense_harness(lin, (x, type_vec))
