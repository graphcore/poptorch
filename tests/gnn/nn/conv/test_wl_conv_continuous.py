# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import torch
from torch_geometric.nn import WLConvContinuous

from conv_utils import conv_harness


def test_wl_conv_cont(dataset):
    in_channels = dataset.num_node_features
    conv = WLConvContinuous()

    lin = torch.nn.Linear(in_channels, 8)
    conv_harness(conv, dataset, post_proc=lin)

    batch = ((dataset.x, None), dataset.edge_index, dataset.edge_weight)
    conv_harness(conv, batch=batch, post_proc=lin)

    x2 = torch.randn(dataset.x.shape)
    batch = ((dataset.x, x2), dataset.edge_index, dataset.edge_weight)
    conv_harness(conv, batch=batch, post_proc=lin)
