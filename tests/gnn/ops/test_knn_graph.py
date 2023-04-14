# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
import helpers

from torch_geometric.nn import knn_graph
from poptorch_geometric.ops.knn_graph import knn_graph as pyg_knn_graph

import poptorch


@pytest.mark.parametrize('flow', ['source_to_target', 'target_to_source'])
def test_knn_graph(flow):
    x = torch.Tensor([[1], [10], [100], [-1], [-10], [-100]])
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    k = 2

    class Model(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return pyg_knn_graph(*args, **kwargs)

    model = poptorch.inferenceModel(Model())

    poptorch_out = model(x, k, batch, True, flow)
    torch_geometric_out = knn_graph(x, k, batch, True, flow)
    pyg_cpu_out = pyg_knn_graph(x, k, batch, True, flow)

    helpers.assert_allclose(actual=poptorch_out, expected=pyg_cpu_out)
    helpers.assert_allclose(actual=poptorch_out, expected=torch_geometric_out)
