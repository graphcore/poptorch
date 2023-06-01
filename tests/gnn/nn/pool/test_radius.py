# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Optional

import torch
import torch_geometric
from torch import Tensor
import poptorch


def to_set(edge_index):
    # pylint: disable=R1721
    return {(i, j) for i, j in edge_index.t().tolist()}


def assert_fn(native_out, poptorch_out):
    poptorch_out = poptorch_out[poptorch_out != -1]
    dim = poptorch_out.size(0) // 2
    poptorch_out = poptorch_out.reshape((2, dim))

    native_out = native_out[native_out != -1]
    dim = native_out.size(0) // 2
    native_out = native_out.reshape((2, dim))

    assert to_set(poptorch_out) == to_set(native_out)


def op_harness(*args, **kwargs):
    class Model(torch.nn.Module):
        def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
            return torch_geometric.nn.radius_graph(x,
                                                   r=2.5,
                                                   batch=batch,
                                                   loop=True)

    native_out = Model()(*args, **kwargs)
    model = poptorch.inferenceModel(Model())
    poptorch_out = model(*args, **kwargs)
    assert_fn(native_out, poptorch_out)


def test_radius_graph():

    x = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float)
    batch = torch.tensor([0, 0, 0, 0])

    op_harness(x, batch)
