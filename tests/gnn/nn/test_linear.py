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


@pytest.mark.skip(reason="TODO(AFS-223)")
def test_hetero_linear():
    x = torch.randn(3, 16)
    type_vec = torch.tensor([0, 1, 2])

    lin = HeteroLinear(16, 32, num_types=3)

    dense_harness(lin, (x, type_vec))
