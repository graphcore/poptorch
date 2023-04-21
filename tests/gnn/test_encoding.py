# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch
from torch_geometric.nn import PositionalEncoding, TemporalEncoding
from gnn.nn.nn_utils import op_harness


def test_positional_encoding():
    encoder = PositionalEncoding(64)

    x = torch.tensor([1.0, 2.0, 3.0])

    op_harness(encoder, [x])


def test_temporal_encoding():
    encoder = TemporalEncoding(64)

    x = torch.tensor([1.0, 2.0, 3.0])

    op_harness(encoder, [x])
