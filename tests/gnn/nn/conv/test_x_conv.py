# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import torch
from torch_geometric.nn import XConv
from torch_geometric.testing import withPackage

from conv_utils import conv_harness


@withPackage('torch_cluster')
def test_x_conv():
    x = torch.randn(8, 16)
    pos = torch.rand(8, 5)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    conv = XConv(16, 32, dim=5, kernel_size=2, dilation=2)

    torch.manual_seed(0)
    # We need to pass very loose atol and rtol here due to TODO(AFS-276)
    conv_harness(conv, batch=(x, pos), atol=0.1, rtol=0.1)
    conv_harness(conv, batch=(x, pos, batch), atol=0.1, rtol=0.1)
