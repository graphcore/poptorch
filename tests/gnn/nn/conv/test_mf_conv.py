# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch
from torch_geometric.nn import MFConv

from conv_utils import conv_harness


def test_mf_conv(dataset, request):
    pytest.skip(
        f'{request.node.nodeid}: AFS-145: Operations using aten::nonzero '
        'are unsupported because the output shape is determined by the '
        'tensor values. The IPU cannot support dynamic output shapes')
    in_channels = dataset.num_node_features
    out_channels = 32

    conv = MFConv(in_channels, out_channels, add_self_loops=False)

    conv_harness(conv, dataset)

    conv = MFConv((in_channels, in_channels),
                  out_channels,
                  add_self_loops=False)

    x2 = torch.randn(dataset.x.shape)
    batch = ((dataset.x, x2), dataset.edge_index)
    conv_harness(conv, batch=batch)

    batch = ((dataset.x, None), dataset.edge_index)
    conv_harness(conv, batch=batch)
