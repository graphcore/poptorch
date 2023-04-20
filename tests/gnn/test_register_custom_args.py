# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import unittest

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import helpers
import poptorch


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16, add_self_loops=False)
        self.conv2 = GCNConv(16, out_channels, add_self_loops=False)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.log_softmax(x, dim=1)

        return x


@unittest.mock.patch.dict("os.environ", helpers.disableSmallModel())
def test_register_custom_parsers(planetoid_cora):
    data = planetoid_cora[0]
    model = GCN(planetoid_cora.num_node_features, planetoid_cora.num_classes)
    model.eval()
    poptorch_model = poptorch.inferenceModel(model)
    result = poptorch_model(data)
    assert result is not None
