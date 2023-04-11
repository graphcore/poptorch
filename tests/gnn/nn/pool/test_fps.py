# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest

import torch
from torch_geometric.nn import fps

import poptorch


class FpsModel(torch.nn.Module):
    def __init__(self, op) -> None:
        super().__init__()
        self.op = op

    def forward(self, *args, **kwargs):
        return self.op(*args, **kwargs)


@pytest.mark.skip(reason="TODO(AFS-266)")
def test_knn():
    x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    batch = torch.tensor([0, 0, 0, 0])

    index_cpu = fps(x, batch, ratio=0.5)

    model = poptorch.inferenceModel(FpsModel(fps))
    index_ipu = model(x, batch, ratio=0.5)

    torch.testing.assert_close(index_ipu, index_cpu)
