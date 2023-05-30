# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch
from torch_geometric.nn import DNAConv

from conv_utils import conv_harness

conv_kwargs_list = [{
    'heads': 4,
    'groups': 8,
}, {
    'heads': 4,
    'groups': 8,
}, {
    'heads': 4,
    'groups': 8,
    'cached': True
}]


@pytest.mark.parametrize('conv_kwargs', conv_kwargs_list)
def test_dna_conv(conv_kwargs):
    channels = 32
    num_layers = 3
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, num_layers, channels))

    conv = DNAConv(channels, dropout=0.0, add_self_loops=False, **conv_kwargs)
    conv_harness(conv, batch=(x, edge_index))
