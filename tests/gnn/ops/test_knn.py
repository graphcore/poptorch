# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
import torch_cluster

import helpers
from poptorch_geometric.ops.knn import knn
import poptorch


def assert_fn(native_out, poptorch_out, x, y):
    row_native, col_native = native_out
    row_poptorch, col_poptorch = poptorch_out

    helpers.assert_allclose(actual=row_poptorch, expected=row_native)
    assert col_native.shape == col_poptorch.shape

    for idx, expected_idx, y_idx in zip(col_native, col_poptorch, row_native):
        if idx != expected_idx:
            helpers.assert_allclose(actual=torch.norm(x[idx] - y[y_idx],
                                                      dim=-1),
                                    expected=torch.norm(x[expected_idx] -
                                                        y[y_idx],
                                                        dim=-1))


def op_harness(op, reference_op, x, y, k, batch_x=None, batch_y=None):

    native_out = reference_op(x, y, k, batch_x, batch_y)

    class Model(torch.nn.Module):
        def forward(self, *args):
            return op(*args)

    model = poptorch.inferenceModel(Model())

    poptorch_out = model(x, y, k, batch_x, batch_y)

    assert_fn(native_out, poptorch_out, x, y)


@pytest.mark.parametrize("with_batch", [True, False])
def test_knn_basic(with_batch):
    pos_x = torch.Tensor([[-1, 0], [0, 0], [1, 0], [-2, 0], [0, 0], [2, 0]])
    pos_y = torch.Tensor([[-1, -1], [1, 1], [-2, -2], [2, 2]])
    k = 2
    if with_batch:
        batch_x = torch.Tensor([0, 0, 0, 1, 1, 1])
        batch_y = torch.Tensor([0, 0, 1, 1])
    else:
        batch_x = None
        batch_y = None

    op_harness(knn, knn, pos_x, pos_y, k, batch_x, batch_y)
    op_harness(knn, torch_cluster.knn, pos_x, pos_y, k, batch_x, batch_y)


def test_knn():
    x = torch.Tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ])
    y = torch.Tensor([
        [1, 0],
        [-1, 0],
    ])

    batch_x = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1])
    batch_y = torch.Tensor([0, 1])
    k = 2

    op_harness(knn, torch_cluster.knn, x, y, k, batch_x, batch_y)
    op_harness(knn, knn, x, y, k, batch_x, batch_y)
    op_harness(knn, torch_cluster.knn, x, y, k)
    op_harness(knn, knn, x, y, k)


def test_knn_batch_skip():
    x = torch.Tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ])
    y = torch.Tensor([
        [1, 0],
        [-1, 0],
    ])

    batch_x = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1])
    batch_y = torch.Tensor([0, 1])
    k = 2

    op_harness(knn, torch_cluster.knn, x, y, k, batch_x, batch_y)
    op_harness(knn, knn, x, y, k, batch_x, batch_y)


@pytest.mark.parametrize("with_batch", [True, False])
def test_knn_override(with_batch):
    pos_x = torch.Tensor([[-1, 0], [0, 0], [1, 0], [-2, 0], [0, 0], [2, 0]])
    pos_y = torch.Tensor([[-1, -1], [1, 1], [-2, -2], [2, 2]])
    k = 2
    if with_batch:
        batch_x = torch.Tensor([0, 0, 0, 1, 1, 1])
        batch_y = torch.Tensor([0, 0, 1, 1])
    else:
        batch_x = None
        batch_y = None

    class Model(torch.nn.Module):
        def forward(self, *args):
            return torch_cluster.knn(*args)

    model = poptorch.inferenceModel(Model())
    poptorch_out = model(pos_x, pos_y, k, batch_x, batch_y)
    native_out = torch_cluster.knn(pos_x, pos_y, k, batch_x, batch_y)
    assert_fn(native_out, poptorch_out, pos_x, pos_y)
