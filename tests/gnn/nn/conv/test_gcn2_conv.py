# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
from torch_geometric.nn import GCN2Conv
from conv_utils import conv_harness

out_channels = 16


def test_gcn2_conv(dataset):
    print(dataset)
    in_channels = dataset.num_node_features
    conv = GCN2Conv(in_channels, alpha=float(0.2), add_self_loops=False)
    x2 = torch.randn_like(dataset.x)
    batch = (dataset.x, x2, dataset.edge_index)
    conv_harness(conv, dataset, batch=batch, num_steps=1)
