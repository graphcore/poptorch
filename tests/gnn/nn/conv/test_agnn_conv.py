# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from torch_geometric.nn import AGNNConv
from conv_utils import conv_harness

conv_kwargs = {"add_self_loops": False}


def test_agnn_conv(dataset):
    conv = AGNNConv(**conv_kwargs)

    conv_harness(conv, dataset)
