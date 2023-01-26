# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
from torch_geometric.nn import LGConv
from conv_utils import conv_harness

out_channels = 16


def test_lg_conv(dataset):
    in_channels = dataset.num_node_features
    conv = LGConv()
    lin = torch.nn.Linear(in_channels, out_channels)

    conv_harness(conv, dataset, post_proc=lin)


def test_lg_edge_weights_conv(dataset):
    in_channels = dataset.num_node_features
    conv = LGConv()
    lin = torch.nn.Linear(in_channels, out_channels)

    batch = (dataset.x, dataset.edge_index, dataset.edge_weight)
    conv_harness(conv, dataset, batch=batch, post_proc=lin)
