# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import torch
from torch_geometric.nn import GeneralConv
from conv_utils import conv_harness

out_channels = 32
num_edge_attr = 16

conv_kwargs_list = [{
    'skip_linear': True
}, {
    'directed_msg': False
}, {
    'heads': 3
}, {
    'attention': True
}, {
    'heads': 3,
    'attention': True
}, {
    'heads': 3,
    'attention': True,
    'attention_type': 'dot_product'
}, {
    'l2_normalize': True
}]


@pytest.mark.parametrize('conv_kwargs', conv_kwargs_list)
def test_general_conv(dataset, conv_kwargs):

    if conv_kwargs.get('attention_type') == 'dot_product':
        pytest.skip("TODO(AFS-37)")

    in_channels = dataset.num_node_features
    conv = GeneralConv(in_channels, out_channels, num_edge_attr, **conv_kwargs)

    e1 = torch.randn(dataset.num_edges, num_edge_attr)

    batch = (dataset.x, dataset.edge_index, e1)
    conv_harness(conv, dataset, batch=batch)
