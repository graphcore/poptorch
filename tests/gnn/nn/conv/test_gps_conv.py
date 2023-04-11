# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch
from torch_geometric.nn import GPSConv, SAGEConv

from conv_utils import conv_harness


@pytest.mark.skip(reason="TODO(AFS-279, AFS-162)")
@pytest.mark.parametrize('norm', [None, 'batch_norm', 'layer_norm'])
def test_gps_conv(norm, dataset):
    in_channels = dataset.num_node_features
    conv = GPSConv(in_channels,
                   conv=SAGEConv(16, 16, add_self_loops=False),
                   heads=4,
                   norm=norm)
    conv.reset_parameters()

    conv_harness(conv, dataset)


@pytest.mark.skip(reason="TODO(AFS-279, AFS-162)")
@pytest.mark.parametrize('norm', [None, 'batch_norm', 'layer_norm'])
def test_gps_conv_with_batch_index_tensor(norm, dataset):
    in_channels = dataset.num_node_features
    conv = GPSConv(in_channels,
                   conv=SAGEConv(16, 16, add_self_loops=False),
                   heads=4,
                   norm=norm)
    conv.reset_parameters()

    batch_index = [
        i > dataset.num_nodes // 2 for i in range(dataset.num_nodes)
    ]
    batch_index = torch.tensor(batch_index, dtype=torch.int64)

    batch = (dataset.x, dataset.edge_index, batch_index)
    conv_harness(conv, batch=batch)
