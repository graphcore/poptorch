# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn import PointTransformerConv
from conv_utils import conv_harness

out_channels = 32


def test_point_transformer_conv(dataset):
    in_channels = dataset.num_node_features
    conv = PointTransformerConv(in_channels,
                                out_channels,
                                add_self_loops=False)

    pos = torch.rand(dataset.num_nodes, 3)

    batch = (dataset.x, pos, dataset.edge_index)
    conv_harness(conv, dataset, batch=batch, atol=1e-4, rtol=1e-3)


def test_point_transformer_nn_conv(dataset):
    in_channels = dataset.num_node_features
    pos_nn = Seq(Lin(3, 16), ReLU(), Lin(16, 32))
    attn_nn = Seq(Lin(32, 32), ReLU(), Lin(32, 32))
    conv = PointTransformerConv(in_channels,
                                out_channels,
                                pos_nn,
                                attn_nn,
                                add_self_loops=False)

    pos = torch.rand(dataset.num_nodes, 3)

    batch = (dataset.x, pos, dataset.edge_index)
    conv_harness(conv, dataset, batch=batch, atol=1e-3, rtol=1e-2)
