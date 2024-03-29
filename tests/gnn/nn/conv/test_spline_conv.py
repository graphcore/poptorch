# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch
from torch_geometric.nn import SplineConv
from torch_geometric.testing import withPackage

from conv_utils import conv_harness


@pytest.mark.parametrize("training", [True, False])
@withPackage('torch_spline_conv')
def test_spline_conv(training):
    if training:
        pytest.skip('reason="TODO(AFS-216, AFS-218)')
    x1 = torch.randn(4, 4)
    x2 = torch.randn(2, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = torch.rand(edge_index[0].size(0), 3)
    conv = SplineConv(4, 32, dim=3, kernel_size=5)

    conv_harness(conv, batch=(x1, edge_index, value), training=training)

    conv = SplineConv((4, 8), 32, dim=3, kernel_size=5)
    batch = ((x1, x2), edge_index, value)
    conv_harness(conv, batch=batch, training=training)

    batch = ((x1, None), edge_index, value, (4, 2))
    conv_harness(conv, batch=batch, training=training)
