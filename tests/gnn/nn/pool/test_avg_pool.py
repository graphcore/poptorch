# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from torch_geometric.data import Batch, Data
from torch_geometric.nn import avg_pool, avg_pool_neighbor_x, avg_pool_x

from pool_utils import pool_harness


def test_avg_pool_x():
    cluster = torch.tensor([0, 1, 0, 1, 2, 2])
    x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1])
    batch_size = int(batch.max().item()) + 1

    out, _ = pool_harness(avg_pool_x, [cluster, x, batch, batch_size, 2])
    assert out.tolist() == [[3, 4], [5, 6], [10, 11], [0, 0]]


@pytest.mark.skip(
    reason="avg_pool uses torch.unique instruction which produces "
    "tensor with dynamic shape. This is not supported for Mk2.")
def test_avg_pool():
    cluster = torch.tensor([0, 1, 0, 1, 2, 2])
    x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    pos = torch.Tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    edge_attr = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    batch = torch.tensor([0, 0, 0, 0, 1, 1])

    data = Batch(x=x,
                 pos=pos,
                 edge_index=edge_index,
                 edge_attr=edge_attr,
                 batch=batch)

    data = pool_harness(avg_pool, [cluster, data, lambda x: x])

    assert data.x.tolist() == [[3, 4], [5, 6], [10, 11]]
    assert data.pos.tolist() == [[1, 1], [2, 2], [4.5, 4.5]]
    assert data.edge_index.tolist() == [[0, 1], [1, 0]]
    assert data.edge_attr.tolist() == [4, 4]
    assert data.batch.tolist() == [0, 0, 1]


@pytest.mark.parametrize('input_type', [Data, Batch])
def test_avg_pool_neighbor_x(input_type):
    if input_type == Batch:
        pytest.skip("TODO(AFS-231, AFS-229, AFS-230)")

    x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1])

    data = input_type(x=x, edge_index=edge_index, batch=batch)

    data = pool_harness(avg_pool_neighbor_x, [data])

    assert data.x.tolist() == [
        [4, 5],
        [4, 5],
        [4, 5],
        [4, 5],
        [10, 11],
        [10, 11],
    ]
    assert data.edge_index.tolist() == edge_index.tolist()
