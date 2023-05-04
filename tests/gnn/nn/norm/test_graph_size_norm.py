# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import torch

from norm_utils import norm_harness

from torch_geometric.nn import GraphSizeNorm


def test_graph_size_norm():
    x = torch.randn(100, 16)
    batch = torch.repeat_interleave(torch.full((10, ), 10, dtype=torch.long))
    batch_size = int(batch.max()) + 1

    norm = GraphSizeNorm()
    assert str(norm) == 'GraphSizeNorm()'

    out = norm_harness(norm, [x, batch, batch_size])
    assert out.size() == (100, 16)
