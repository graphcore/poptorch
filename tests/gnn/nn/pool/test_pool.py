# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Optional

import pytest
import torch
from torch import Tensor

from torch_geometric.nn import radius_graph

from pool_utils import op_harness


def test_radius_graph(request):
    pytest.skip(
        f"{request.node.nodeid}: Error: "
        "torch_geometric/nn/pool/__init__.py:210: RuntimeError: "
        "x.device().is_cpu() INTERNAL ASSERT FAILED at "
        "\"csrc/cpu/radius_cpu.cpp\":12, please report a bug to PyTorch. x "
        "must be CPU tensor. Will be enabled after AFS-147 is fixed.")

    class Net(torch.nn.Module):
        def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
            return radius_graph(x, r=2.5, batch=batch, loop=False)

    x = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float)
    batch = torch.tensor([0, 0, 0, 0])

    op_harness(Net, [x, batch])
