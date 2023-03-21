# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import torch
from torch_geometric import seed_everything
from torch_geometric.nn import MLP, PointGNNConv

from conv_utils import conv_harness


def test_pointgnn_conv():
    seed_everything(42)
    x = torch.rand(6, 8)
    pos = torch.rand(6, 3)
    edge_index = torch.tensor([[0, 1, 1, 1, 2, 5], [1, 2, 3, 4, 3, 4]])

    conv = PointGNNConv(
        mlp_h=MLP([8, 16, 3], norm='LayerNorm'),
        mlp_f=MLP([3 + 8, 16, 8], norm='LayerNorm'),
        mlp_g=MLP([8, 16, 8], norm='LayerNorm'),
    )

    batch = (x, pos, edge_index)
    conv_harness(conv, batch=batch)
