# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
from torch_geometric.nn import DenseGCNConv, DenseGraphConv, DenseGINConv, DenseSAGEConv
import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from dense_utils import dense_harness


@pytest.mark.parametrize(
    "conv_fn", [DenseGCNConv, DenseGraphConv, DenseGINConv, DenseSAGEConv])
def test_dense_convs(conv_fn):
    channels = 16
    if conv_fn is DenseGINConv:
        nn = Seq(Lin(channels, channels), ReLU(), Lin(channels, channels))
        conv = conv_fn(nn)
    else:
        conv = conv_fn(channels, channels)
    x = torch.randn((5, channels))
    x = torch.cat([x, x.new_zeros(1, channels)], dim=0).view(2, 3, channels)
    adj = torch.Tensor([
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
    ])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)

    batch = (x, adj, mask)
    dense_out = conv(*batch)
    assert dense_out.size() == (2, 3, channels)
    assert dense_out[1, 2].abs().sum().item() == 0

    dense_harness(conv, batch)


@pytest.mark.parametrize(
    "conv_fn", [DenseGCNConv, DenseGraphConv, DenseGINConv, DenseSAGEConv])
def test_dense_convs_with_broadcasting(conv_fn):
    batch_size, num_nodes, channels = 8, 3, 16
    if conv_fn is DenseGINConv:
        nn = Seq(Lin(channels, channels), ReLU(), Lin(channels, channels))
        conv = conv_fn(nn)
    else:
        conv = conv_fn(channels, channels)

    x = torch.randn(batch_size, num_nodes, channels)
    adj = torch.Tensor([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])

    assert conv(x, adj).size() == (batch_size, num_nodes, channels)
    mask = torch.tensor([1, 1, 1], dtype=torch.bool)
    batch = (x, adj, mask)
    assert conv(*batch).size() == (batch_size, num_nodes, channels)

    dense_harness(conv, batch)
