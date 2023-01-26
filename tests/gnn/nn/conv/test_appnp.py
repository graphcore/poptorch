# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
from torch_geometric.nn import APPNP
from conv_utils import conv_harness

out_channels = 16
conv_kwargs = {"add_self_loops": False}


def test_appnp(dataset):
    in_channels = dataset.num_node_features
    lin = torch.nn.Linear(in_channels, out_channels)
    conv = APPNP(K=10, alpha=0.1, dropout=0.0, **conv_kwargs)

    conv_harness(conv, dataset, post_proc=lin)
