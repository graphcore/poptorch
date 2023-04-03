# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import helpers

import torch
from torch_geometric.nn import knn_interpolate

import poptorch
import poptorch_geometric  # pylint: disable=unused-import


@pytest.mark.skip()
def test_knn_interpolate():
    x = torch.Tensor([[1], [10], [100], [-1], [-10], [-100]])
    pos_x = torch.Tensor([[-1, 0], [0, 0], [1, 0], [-2, 0], [0, 0], [2, 0]])
    pos_y = torch.Tensor([[-1, -1], [1, 1], [-2, -2], [2, 2]])
    batch_x = torch.tensor([0, 0, 0, 1, 1, 1])
    batch_y = torch.tensor([0, 0, 1, 1])
    k = 2

    class Model(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return knn_interpolate(*args, **kwargs)

    model = poptorch.inferenceModel(Model())

    poptorch_out = model(x, pos_x, pos_y, batch_x, batch_y, k)
    torch_geometric_out = knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k)

    helpers.assert_allclose(actual=poptorch_out, expected=torch_geometric_out)
