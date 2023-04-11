# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest

import torch
from torch_geometric.nn import knn, knn_graph

import helpers
import poptorch


class KnnModel(torch.nn.Module):
    def __init__(self, op) -> None:
        super().__init__()
        self.op = op

    def forward(self, *args, **kwargs):
        return self.op(*args, **kwargs)


@pytest.mark.skip(reason="TODO(AFS-291)")
def test_knn():
    x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    batch_x = torch.tensor([0, 0, 0, 0])
    y = torch.Tensor([[-1, 0], [1, 0]])
    batch_y = torch.tensor([0, 0])

    assign_index_cpu = knn(x, y, 2, batch_x, batch_y)

    model = poptorch.inferenceModel(KnnModel(knn))
    assign_index_ipu = model(x, y, 2, batch_x, batch_y)

    helpers.assert_allclose(actual=assign_index_ipu, expected=assign_index_cpu)


@pytest.mark.skip(reason="TODO(AFS-273)")
def test_knn_graph():
    x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    batch = torch.tensor([0, 0, 0, 0])

    edge_index_cpu = knn_graph(x, k=2, batch=batch, loop=True)
    print(edge_index_cpu)
    model = poptorch.inferenceModel(KnnModel(knn_graph))
    edge_index_ipu = model(x, k=2, batch=batch, loop=True)

    helpers.assert_allclose(actual=edge_index_cpu, expected=edge_index_ipu)
