# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch
from torch_geometric.nn import HEATConv

from conv_utils import conv_harness


@pytest.mark.skip(reason="TODO(AFS-223)")
@pytest.mark.parametrize('concat', [True, False])
def test_heat_conv(concat):
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_attr = torch.randn((4, 2))
    node_type = torch.tensor([0, 0, 1, 2])
    edge_type = torch.tensor([0, 2, 1, 2])

    conv = HEATConv(in_channels=8,
                    out_channels=16,
                    num_node_types=3,
                    num_edge_types=3,
                    edge_type_emb_dim=5,
                    edge_dim=2,
                    edge_attr_emb_dim=6,
                    heads=2,
                    concat=concat,
                    add_self_loops=False)

    conv_harness(conv, batch=(x, edge_index, node_type, edge_type, edge_attr))
