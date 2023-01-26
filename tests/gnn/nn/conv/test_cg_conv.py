# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
from torch_geometric.nn import CGConv
from conv_utils import conv_harness


@pytest.mark.parametrize('batch_norm', [False])
def test_cg_conv(dataset, batch_norm):
    in_channels = dataset.num_node_features
    conv = CGConv(in_channels, batch_norm=batch_norm)

    conv_harness(conv, dataset)
