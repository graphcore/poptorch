# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from collections import defaultdict

import pytest

from torch_geometric.nn import HGTConv

from conv_utils import hetero_conv_harness, random_heterodata


@pytest.mark.skip(reason="TODO(AFS-309)")
def test_hgt_conv_same_dimensions():
    in_channels = defaultdict(lambda: 16)

    data, _ = random_heterodata(in_channels)

    conv = HGTConv(in_channels['author'],
                   in_channels['paper'],
                   metadata=data.metadata(),
                   heads=2)
    hetero_conv_harness(conv, data, 'author')


@pytest.mark.skip(reason="TODO(AFS-309)")
def test_hgt_conv_different_dimensions():
    in_channels = defaultdict(lambda: 16)
    in_channels['paper'] = 32

    data, _ = random_heterodata(in_channels)

    conv = HGTConv(in_channels=in_channels,
                   out_channels=32,
                   metadata=data.metadata(),
                   heads=2)

    hetero_conv_harness(conv, data, 'author')


@pytest.mark.skip(reason="TODO(AFS-309)")
def test_hgt_conv_lazy():
    in_channels = defaultdict(lambda: 16)
    in_channels['paper'] = 32

    data, _ = random_heterodata(in_channels)

    conv = HGTConv(-1, 32, metadata=data.metadata(), heads=2)

    _ = conv(data.x_dict, data.edge_index_dict)
    hetero_conv_harness(conv, data, 'author')
