# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
from torch_geometric.nn import SignedConv
from conv_utils import conv_harness

out_channels = 16


def test_signed_conv(dataset):

    in_channels = dataset.num_node_features

    class Convs(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SignedConv(in_channels,
                                    out_channels,
                                    first_aggr=True,
                                    add_self_loops=False)

            self.conv2 = SignedConv(out_channels,
                                    32,
                                    first_aggr=False,
                                    add_self_loops=False)

        def forward(self, x, pos_edge_index, neg_edge_index):
            x = self.conv1(x, pos_edge_index, neg_edge_index)
            x = self.conv2(x, pos_edge_index, neg_edge_index)
            return x

    conv = Convs()

    batch = (dataset.x, dataset.edge_index, dataset.edge_index)
    conv_harness(conv, dataset, batch=batch)
