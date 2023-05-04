# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import torch

from norm_utils import norm_harness

from torch_geometric.nn import GraphNorm


def test_graph_norm():
    torch.manual_seed(42)
    x = torch.randn(200, 16)
    batch = torch.arange(4).view(-1, 1).repeat(1, 50).view(-1)
    batch_size = int(batch.max() + 1)

    norm = GraphNorm(16)

    norm_harness(norm, [x])
    norm_harness(norm, [x, batch, batch_size])
