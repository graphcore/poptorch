# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from torch_geometric.nn import FeaStConv
from conv_utils import conv_harness

out_channels = 32
conv_kwargs = {"add_self_loops": False}


def test_feast_conv(dataset):
    in_channels = dataset.num_node_features
    conv = FeaStConv(in_channels, out_channels, heads=2, **conv_kwargs)

    conv_harness(conv, dataset)
