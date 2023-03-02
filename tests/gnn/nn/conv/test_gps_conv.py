# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch
from torch_geometric.nn import GPSConv, SAGEConv

from conv_utils import conv_harness


@pytest.mark.parametrize('norm', [None, 'batch_norm', 'layer_norm'])
def test_gps_conv(norm, dataset, request):
    pytest.skip(
        f'{request.node.nodeid}: AFS-145: Operations using aten::nonzero '
        'are unsupported because the output shape is determined by the '
        'tensor values. The IPU cannot support dynamic output shapes')

    in_channels = dataset.num_node_features
    conv = GPSConv(in_channels,
                   conv=SAGEConv(16, 16, add_self_loops=False),
                   heads=4,
                   norm=norm)
    conv.reset_parameters()

    conv_harness(conv, dataset)


@pytest.mark.parametrize('norm', [None, 'batch_norm', 'layer_norm'])
def test_gps_conv_with_batch_index_tensor(norm, dataset, request):
    pytest.skip(
        f'{request.node.nodeid}: AFS-144: Could not run '
        'aten::_local_scalar_dense with arguments from the Meta backend.')

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
