# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
from torch_geometric.nn import SuperGATConv
from conv_utils import conv_harness

out_channels = 16


@pytest.mark.skip(reason="TODO(AFS-36)")
@pytest.mark.parametrize('att_type', ['MX', 'SD'])
def test_supergat_conv(dataset, att_type):
    in_channels = dataset.num_node_features
    conv = SuperGATConv(in_channels,
                        out_channels,
                        heads=2,
                        attention_type=att_type,
                        neg_sample_ratio=1.0,
                        edge_sample_ratio=1.0,
                        add_self_loops=False)

    conv_harness(conv, dataset)
