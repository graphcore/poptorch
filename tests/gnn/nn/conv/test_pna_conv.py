# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
from torch_geometric.nn import PNAConv
from conv_utils import conv_harness

out_channels = 16

aggregators = ['sum', 'mean', 'min', 'max', 'var', 'std']
scalers = [
    'identity', 'amplification', 'attenuation', 'linear', 'inverse_linear'
]


def test_pna_conv(dataset):
    in_channels = dataset.num_node_features
    deg = PNAConv.get_degree_histogram([dataset])

    conv = PNAConv(in_channels,
                   out_channels,
                   aggregators,
                   scalers,
                   deg=deg,
                   edge_dim=3,
                   towers=4)

    value = torch.rand(dataset.num_edges, 3)
    batch = (dataset.x, dataset.edge_index, value)
    conv_harness(conv, dataset, batch=batch)
