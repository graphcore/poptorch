# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
import torch_geometric
import helpers
import poptorch

from poptorch_geometric.ops.to_dense_batch import to_dense_batch


def op_harness(reference_op, *args, **kwargs):
    class Model(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return torch_geometric.utils.to_dense_batch(*args, **kwargs)

    model = poptorch.inferenceModel(Model())

    poptorch_out = model(*args, **kwargs)

    native_out = reference_op(*args, **kwargs)

    helpers.assert_allclose(actual=poptorch_out, expected=native_out)


def test_basic():
    x = torch.arange(12).view(6, 2)

    op_harness(to_dense_batch, x)


def test_batch_size_not_set():
    x = torch.arange(12).view(6, 2)
    batch = torch.tensor([0, 0, 1, 2, 2, 2])

    with pytest.raises(RuntimeError):
        op_harness(to_dense_batch, x, batch)


def test_batch_size_set():
    x = torch.arange(12).view(6, 2)
    batch = torch.tensor([0, 0, 1, 2, 2, 2])

    with pytest.raises(RuntimeError):
        op_harness(to_dense_batch, x, batch)


def test_batch_size_and_max_num_nodes_set():
    x = torch.arange(12).view(6, 2)
    batch = torch.tensor([0, 0, 1, 2, 2, 2])
    batch_size = int(batch.max()) + 1
    max_num_nodes = 11

    op_harness(to_dense_batch,
               x,
               batch,
               max_num_nodes=max_num_nodes,
               batch_size=batch_size)
