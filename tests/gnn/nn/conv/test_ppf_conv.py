# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn import PPFConv
from conv_utils import conv_harness


def test_ppf_conv(dataset):

    local_nn = Seq(Lin(16 + 4, 32), ReLU(), Lin(32, 32))
    global_nn = Seq(Lin(32, 32))
    conv = PPFConv(local_nn, global_nn, add_self_loops=False)

    pos = torch.rand(dataset.num_nodes, 3)
    n = F.normalize(torch.rand(dataset.num_nodes, 3), dim=-1)

    batch = (dataset.x, pos, n, dataset.edge_index)
    conv_harness(conv, dataset, batch=batch)


def test_ppf2_conv(dataset):

    local_nn = Seq(Lin(16 + 4, 32), ReLU(), Lin(32, 32))
    global_nn = Seq(Lin(32, 32))
    conv = PPFConv(local_nn, global_nn, add_self_loops=False)

    pos1 = torch.rand(dataset.num_nodes, 3)
    pos2 = torch.rand(dataset.num_nodes, 3)
    n1 = F.normalize(torch.rand(dataset.num_nodes, 3), dim=-1)
    n2 = F.normalize(torch.rand(dataset.num_nodes, 3), dim=-1)

    batch = (dataset.x, (pos1, pos2), (n1, n2), dataset.edge_index)
    conv_harness(conv, dataset, batch=batch)
