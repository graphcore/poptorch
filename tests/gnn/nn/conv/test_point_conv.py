# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn import PointNetConv
from conv_utils import conv_harness

out_channels = 16


def test_point_net_conv(dataset):

    local_nn = Seq(Lin(16 + 3, 32), ReLU(), Lin(32, 32))
    global_nn = Seq(Lin(32, 32))
    conv = PointNetConv(local_nn, global_nn, add_self_loops=False)

    pos = torch.rand(dataset.num_nodes, 3)
    batch = (dataset.x, pos, dataset.edge_index)
    conv_harness(conv, dataset, batch=batch)


def test_point2_net_conv(dataset):

    local_nn = Seq(Lin(16 + 3, 32), ReLU(), Lin(32, 32))
    global_nn = Seq(Lin(32, 32))
    conv = PointNetConv(local_nn, global_nn, add_self_loops=False)

    pos1 = torch.rand(dataset.num_nodes, 3)
    pos2 = torch.rand(dataset.num_nodes, 3)

    batch = (dataset.x, (pos1, pos2), dataset.edge_index)
    conv_harness(conv, dataset, batch=batch)
