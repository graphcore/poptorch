# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import torch
from torch_geometric.nn import GravNetConv
from torch_geometric.testing import withPackage

from conv_utils import conv_harness


@withPackage('torch_cluster')
def test_gravnet_conv(dataset):
    in_channels = dataset.num_node_features
    out_channels = 32
    conv = GravNetConv(in_channels,
                       out_channels,
                       space_dimensions=4,
                       propagate_dimensions=8,
                       k=2,
                       add_self_loops=False)
    conv_harness(conv, batch=(dataset.x, ))

    num_nodes = dataset.num_nodes
    batch_index = [1 if i > num_nodes // 2 else 0 for i in range(num_nodes)]
    conv_harness(conv, batch=(dataset.x, batch_index))

    x2 = torch.randn_like(dataset.x)
    conv_harness(conv, batch=((dataset.x, x2), ), atol=5e-05, rtol=0.001)
    conv_harness(conv,
                 batch=((dataset.x, x2), (torch.Tensor(batch_index),
                                          torch.Tensor(batch_index))),
                 atol=5e-03,
                 rtol=0.1)
