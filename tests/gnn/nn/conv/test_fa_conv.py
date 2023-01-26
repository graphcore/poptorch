# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from torch_geometric.nn import FAConv
from conv_utils import conv_harness

conv_kwargs = {"add_self_loops": False}


def test_fa_conv(dataset):
    in_channels = dataset.num_node_features
    conv = FAConv(in_channels, eps=1.0, **conv_kwargs)
    batch = (dataset.x, dataset.x, dataset.edge_index)

    conv_harness(conv, dataset, batch=batch)
