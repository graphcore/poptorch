# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch
from torch_geometric.nn import FastRGCNConv, RGCNConv
from conv_utils import conv_harness

out_channels = 16


@pytest.mark.parametrize('rgcn', [FastRGCNConv, RGCNConv])
def test_rgcn_conv(rgcn):
    if rgcn == RGCNConv:
        pytest.skip("RGCNConv uses dynamic shapes")

    in_channels = 4
    out_channels = 32
    num_relations = 4
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [0, 0, 1, 0, 1, 1]])
    edge_type = torch.tensor([0, 1, 1, 0, 0, 1])
    conv = rgcn(in_channels,
                out_channels,
                num_relations,
                num_bases=15,
                add_self_loops=False)

    batch = (None, edge_index, edge_type)
    conv_harness(conv, edge_index_max=3, batch=batch)
