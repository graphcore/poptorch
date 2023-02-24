# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import torch

from norm_utils import norm_harness

from torch_geometric.nn import MessageNorm


def test_message_norm():
    norm = MessageNorm(learn_scale=True)
    assert str(norm) == 'MessageNorm(learn_scale=True)'
    x = torch.randn(100, 16)
    msg = torch.randn(100, 16)

    out = norm_harness(norm, [x, msg])
    assert out.size() == (100, 16)

    norm = MessageNorm(learn_scale=False)
    assert str(norm) == 'MessageNorm(learn_scale=False)'
    out = norm_harness(norm, [x, msg])
    assert out.size() == (100, 16)
