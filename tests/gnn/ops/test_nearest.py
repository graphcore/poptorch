# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
import torch_cluster

from poptorch_geometric.ops.nearest import nearest
import poptorch


def op_harness(op, reference_op, x, y, batch_x=None, batch_y=None):
    native_out = reference_op(x, y, batch_x, batch_y)

    class Model(torch.nn.Module):
        def forward(self, *args):
            return op(*args)

    model = poptorch.inferenceModel(Model())
    poptorch_out = model(x, y, batch_x, batch_y)

    assert all(native_out == poptorch_out)


@pytest.mark.parametrize('dtype', [torch.half, torch.float, torch.double])
def test_nearest(dtype):
    x = torch.tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
        [-2, -2],
        [-2, +2],
        [+2, +2],
        [+2, -2],
    ],
                     dtype=dtype)
    y = torch.tensor([
        [-1, 0],
        [+1, 0],
        [-2, 0],
        [+2, 0],
    ], dtype=dtype)

    batch_x = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    batch_y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    op_harness(nearest, nearest, x, y, batch_x, batch_y)
    op_harness(nearest, torch_cluster.nearest, x, y, batch_x, batch_y)

    op_harness(nearest, nearest, x, y)
    op_harness(nearest, torch_cluster.nearest, x, y)

    # Invalid input: instance 1 only in batch_x
    batch_x = [0, 0, 0, 0, 1, 1, 1, 1]
    batch_y = [0, 0, 0, 0]
    with pytest.raises(ValueError):
        op_harness(nearest, nearest, x, y, batch_x, batch_y)

    # Invalid input: instance 1 only in batch_x (implicitly as batch_y=None)
    with pytest.raises(ValueError):
        op_harness(nearest, nearest, x, y, batch_x, None)

    # Invalid input: instance 2 only in batch_x
    # (i.e.instance in the middle missing)
    batch_x = [0, 0, 1, 1, 2, 2, 3, 3]
    batch_y = [0, 1, 3, 3]
    with pytest.raises(ValueError):
        op_harness(nearest, nearest, x, y, batch_x, batch_y)

    # Invalid input: batch_x unsorted
    batch_x = [0, 0, 1, 0, 0, 0, 0]
    batch_y = [0, 0, 1, 1]
    with pytest.raises(ValueError):
        op_harness(nearest, nearest, x, y, batch_x, batch_y)

    # Invalid input: batch_y unsorted
    batch_x = [0, 0, 0, 0, 1, 1, 1, 1]
    batch_y = [0, 0, 1, 0]
    with pytest.raises(ValueError):
        op_harness(nearest, nearest, x, y, batch_x, batch_y)
