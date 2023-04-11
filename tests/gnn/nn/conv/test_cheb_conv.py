# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import torch
import pytest
from torch_geometric.nn import ChebConv

from conv_utils import conv_harness


@pytest.mark.skip(
    reason="ChebConv won't work, because algorithm requires removing "
    "self loops and we are adding self loops to ensure that "
    "tensors have fixed size.")
def test_cheb_conv(dataset):
    in_channels = dataset.num_node_features
    out_channels = 32
    conv = ChebConv(in_channels, out_channels, K=3, add_self_loops=False)
    conv_harness(conv, dataset)

    batch = (dataset.x, dataset.edge_index, dataset.edge_weight)
    conv_harness(conv, batch=batch)

    batch = (dataset.x, dataset.edge_index, dataset.edge_weight, None, 3.0)
    conv_harness(conv, batch=batch)

    num_nodes = dataset.num_nodes
    batch_mask = [int(i > num_nodes // 2) for i in range(num_nodes)]
    batch_mask = torch.tensor(batch_mask)
    lambda_max = torch.tensor([2.0, 3.0])
    batch = (dataset.x, dataset.edge_index, dataset.edge_weight, batch_mask,
             lambda_max)
    conv_harness(conv, batch=batch)
