# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
from torch_geometric.nn import PDNConv
from conv_utils import conv_harness

out_channels = 16


def test_pdn_conv(dataset):
    in_channels = dataset.num_node_features
    conv = PDNConv(in_channels,
                   out_channels,
                   edge_dim=8,
                   hidden_channels=128,
                   add_self_loops=False)

    edge_attr = torch.randn(dataset.num_edges, 8)
    batch = (dataset.x, dataset.edge_index, edge_attr)
    conv_harness(conv, dataset, batch=batch)
