# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch
from torch_geometric.nn import GMMConv

from conv_utils import conv_harness


@pytest.mark.parametrize('separate_gaussians', [True, False])
def test_gmm_conv(separate_gaussians, dataset):
    in_channels = dataset.num_node_features
    conv = GMMConv(in_channels,
                   32,
                   dim=3,
                   kernel_size=25,
                   separate_gaussians=separate_gaussians,
                   add_self_loops=False)
    value = torch.rand(dataset.num_edges, 3)
    batch = (dataset.x, dataset.edge_index, value)
    conv_harness(conv, batch=batch)


@pytest.mark.parametrize('separate_gaussians', [True, False])
def test_gmm_conv_bipartite(separate_gaussians, dataset):

    in_channels = dataset.num_node_features
    conv = GMMConv((in_channels, in_channels),
                   32,
                   dim=3,
                   kernel_size=5,
                   separate_gaussians=separate_gaussians,
                   add_self_loops=False)
    value = torch.rand(dataset.num_edges, 3)
    x2 = torch.randn(dataset.x.shape)
    batch = ((dataset.x, x2), dataset.edge_index, value)
    conv_harness(conv, batch=batch)

    batch = ((dataset.x, None), dataset.edge_index, value)
    conv_harness(conv, batch=batch)
