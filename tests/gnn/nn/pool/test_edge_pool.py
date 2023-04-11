# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from torch_geometric.nn import EdgePooling
from torch_geometric.utils import scatter

from pool_utils import pool_harness


def test_compute_edge_score_softmax():
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    raw = torch.randn(edge_index.size(1))
    e = pool_harness(EdgePooling.compute_edge_score_softmax,
                     [raw, edge_index, 6])
    assert torch.all(e >= 0) and torch.all(e <= 1)

    # Test whether all incoming edge scores sum up to one.
    assert torch.allclose(scatter(e, edge_index[1], reduce='sum'),
                          torch.Tensor([1, 1, 1, 1, 1, 1]))


def test_compute_edge_score_tanh():
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    raw = torch.randn(edge_index.size(1))
    e = pool_harness(EdgePooling.compute_edge_score_tanh, [raw, edge_index, 6])
    assert torch.all(e >= -1) and torch.all(e <= 1)
    assert torch.all(torch.argsort(raw) == torch.argsort(e))


def test_compute_edge_score_sigmoid():
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    raw = torch.randn(edge_index.size(1))
    e = pool_harness(EdgePooling.compute_edge_score_sigmoid,
                     [raw, edge_index, 6])
    assert torch.all(e >= 0) and torch.all(e <= 1)
    assert torch.all(torch.argsort(raw) == torch.argsort(e))


@pytest.mark.skip(
    reason="Currently not possible to run on Mk2 due to the algorithm "
    "used in the __merge_edges__ function")
def test_edge_pooling():
    x = torch.Tensor([[0], [1], [2], [3], [4], [5], [-1]])
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4, 0]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 0])

    op = EdgePooling(in_channels=1)
    assert str(op) == 'EdgePooling(1)'

    # Setting parameters fixed so we can test the expected outcome:
    op.lin.weight.data.fill_(1.)
    op.lin.bias.data.fill_(0.)

    # Test pooling:
    new_x, new_edge_index, new_batch, _ = pool_harness(op,
                                                       [x, edge_index, batch])

    assert new_x.size(0) == new_batch.size(0) == 4
    assert new_edge_index.tolist() == [[0, 1, 1, 2, 2, 3], [0, 1, 2, 1, 2, 2]]
    assert new_batch.tolist() == [1, 0, 0, 0]

    # Test edge cases.
    x = torch.Tensor([[0], [1], [2], [3], [4], [5]])
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1])
    new_x, new_edge_index, new_batch, _ = pool_harness(op,
                                                       [x, edge_index, batch])

    assert new_x.size(0) == new_batch.size(0) == 3
    assert new_batch.tolist() == [1, 0, 0]
    assert new_edge_index.tolist() == [[0, 1, 1, 2, 2], [0, 1, 2, 1, 2]]


def test_edge_unpooling():

    x = torch.Tensor([[0], [1], [2], [3], [4], [5], [-1]])
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4, 0]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 0])

    op = EdgePooling(in_channels=1)
    assert str(op) == 'EdgePooling(1)'

    # Setting parameters fixed so we can test the expected outcome:
    op.lin.weight.data.fill_(1.)
    op.lin.bias.data.fill_(0.)

    # Test pooling:
    new_x, _, _, unpool_info = op(x, edge_index, batch)

    out = pool_harness(op.unpool, [new_x, unpool_info])
    assert out[0].size() == x.size()
    assert out[0].tolist() == [[1], [1], [5], [5], [9], [9], [-1]]
    assert torch.equal(out[1], edge_index)
    assert torch.equal(out[2], batch)
