# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch
from torch_geometric.nn import WLConv

from conv_utils import conv_harness


@pytest.mark.skip(reason="Algorithm requires reading tensors which "
                  "are placed on the IPU.")
def test_wl_conv():
    x = torch.tensor([1, 0, 0, 1])
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    conv = WLConv()
    _ = conv(x, edge_index)
    conv_harness(conv, batch=(x, edge_index), training=False)
