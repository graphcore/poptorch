# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from torch_geometric.data import Batch, Data
from torch_geometric.nn import max_pool, max_pool_neighbor_x, max_pool_x

from pool_utils import pool_harness


def test_max_pool_x(request):
    pytest.skip(
        f"{request.node.nodeid}: Error: 'In "
        "poptorch/source/dispatch_tracer/TypeInferenceHandler.cpp:26: "
        "'poptorch_cpp_error': Type inference failed for aten::_unique2 "
        "because the operator doesn't have an implementation for the Meta "
        "backend'. Will be enabled after AFS-136 is fixed.")

    cluster = torch.tensor([0, 1, 0, 1, 2, 2])
    x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1])

    out = pool_harness(max_pool_x, [cluster, x, batch])
    assert out[0].tolist() == [[5, 6], [7, 8], [11, 12]]
    assert out[1].tolist() == [0, 0, 1]


def test_max_pool_x_size_2(request):
    pytest.skip(
        f"{request.node.nodeid}: Error: 'NotImplementedError: Could "
        "not run 'aten::_local_scalar_dense' with arguments from the 'Meta' "
        "backend'. Will be enabled after AFS-144 is fixed.")

    cluster = torch.tensor([0, 1, 0, 1, 2, 2])
    x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1])

    out, _ = pool_harness(max_pool_x, [cluster, x, batch, 2])
    assert out.tolist() == [[5, 6], [7, 8], [11, 12], [0, 0]]


def test_max_pool(request):
    pytest.skip(f"{request.node.nodeid}: Error: "
                "'poptorch_geometric/types.py:51 AssertionError: Field `ptr` "
                "missing'. Will be enabled after AFS-137 is fixed.")

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

    data = pool_harness(max_pool, [cluster, data, lambda x: x])

    assert data.x.tolist() == [[5, 6], [7, 8], [11, 12]]
    assert data.pos.tolist() == [[1, 1], [2, 2], [4.5, 4.5]]
    assert data.edge_index.tolist() == [[0, 1], [1, 0]]
    assert data.edge_attr.tolist() == [4, 4]
    assert data.batch.tolist() == [0, 0, 1]


@pytest.mark.parametrize('input_type', [Data, Batch])
def test_max_pool_neighbor_x(request, input_type):
    if input_type == Batch:
        pytest.skip(
            f"{request.node.nodeid}: Error: "
            "'poptorch_geometric/types.py:51 AssertionError: Field `ptr` "
            "missing'. Will be enabled after AFS-137 is fixed.")

    x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1])

    data = input_type(x=x, edge_index=edge_index, batch=batch)
    data = pool_harness(max_pool_neighbor_x, [data])

    assert data.x.tolist() == [
        [7, 8],
        [7, 8],
        [7, 8],
        [7, 8],
        [11, 12],
        [11, 12],
    ]
    assert data.edge_index.tolist() == edge_index.tolist()
