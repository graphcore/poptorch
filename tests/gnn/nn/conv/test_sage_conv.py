# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from torch_geometric.nn import SAGEConv
from conv_utils import conv_harness

out_channels = 16

aggregators = ['sum', 'mean', 'min', 'max', 'var', 'std']


def test_sage_conv(dataset):
    in_channels = dataset.num_node_features

    conv = SAGEConv(in_channels,
                    out_channels,
                    aggr=aggregators,
                    normalize=True,
                    root_weight=True,
                    project=True,
                    bias=True)

    conv_harness(conv, dataset)
