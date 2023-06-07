# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
import torch_geometric
from torch_geometric.utils import to_dense_batch

import helpers
import poptorch


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

    op_harness(to_dense_batch, x, batch_size=1, max_num_nodes=11)


def test_batch_size_not_set():
    x = torch.arange(12).view(6, 2)
    batch = torch.tensor([0, 0, 1, 2, 2, 2])

    with pytest.raises(
            ValueError,
            match=
            "Dynamic shapes disabled. Argument 'batch_size' needs to be set"):
        op_harness(to_dense_batch, x, batch)


def test_batch_size_set():
    x = torch.arange(12).view(6, 2)
    batch = torch.tensor([0, 0, 1, 2, 2, 2])

    with pytest.raises(
            ValueError,
            match=
            "Dynamic shapes disabled. Argument 'max_num_nodes' needs to be set"
    ):
        op_harness(to_dense_batch, x, batch, batch_size=3)


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
