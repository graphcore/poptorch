# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from collections import OrderedDict
from torch.nn import ReLU
from torch_geometric.nn import Sequential, GCNConv, Linear

from conv.conv_utils import conv_harness

conv_kwargs = {"add_self_loops": False}


def test_sequential(dataset):
    out_channels = in_channels = dataset.num_node_features

    model = Sequential('x, edge_index', [
        (GCNConv(in_channels, 64, **conv_kwargs), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (GCNConv(64, 64, **conv_kwargs), 'x, edge_index -> x'),
        ReLU(inplace=True),
        Linear(64, out_channels),
    ])

    conv_harness(model, dataset)


def test_sequential_with_ordered_dict(dataset):
    in_channels = dataset.num_node_features

    model = Sequential('x, edge_index',
                       modules=OrderedDict([
                           ('conv1', (GCNConv(in_channels, 32, **conv_kwargs),
                                      'x, edge_index -> x')),
                           ('conv2', (GCNConv(32, 64, **conv_kwargs),
                                      'x, edge_index -> x')),
                       ]))

    conv_harness(model, dataset)
