# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
from torch_geometric.nn import GCNConv
from conv_utils import conv_harness

out_channels = 32
conv_kwargs = {'add_self_loops': False}


@pytest.mark.parametrize('flow', ['source_to_target', 'target_to_source'])
def test_gcn_conv(dataset, flow):
    in_channels = dataset.num_node_features
    conv = GCNConv(in_channels, out_channels, flow, **conv_kwargs)

    conv_harness(conv, dataset)
