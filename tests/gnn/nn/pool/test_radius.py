# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Optional

import pytest
import torch
from torch import Tensor

from torch_geometric.nn import radius_graph

from pool_utils import pool_harness


@pytest.mark.skip(reason="TODO(AFS-263, AFS-264)")
def test_radius_graph():
    class Net(torch.nn.Module):
        def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
            return radius_graph(x, r=2.5, batch=batch, loop=False)

    x = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float)
    batch = torch.tensor([0, 0, 0, 0])

    pool_harness(Net, [x, batch])
