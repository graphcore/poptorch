# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch
from torch_geometric.nn import HypergraphConv

from conv_utils import conv_harness


def test_hypergraph_conv_with_more_nodes_than_edges(request):
    pytest.skip(
        f'{request.node.nodeid}: AFS-144: Could not run '
        'aten::_local_scalar_dense with arguments from the Meta backend.')

    in_channels, out_channels = (16, 32)
    hyperedge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [0, 1, 0, 1, 0, 1]])
    hyperedge_weight = torch.tensor([1.0, 0.5])
    num_nodes = hyperedge_index[0].max().item() + 1
    num_edges = hyperedge_index[1].max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    hyperedge_attr = torch.randn((num_edges, in_channels))

    conv = HypergraphConv(in_channels, out_channels, add_self_loops=False)

    conv_harness(conv, batch=(x, hyperedge_index))

    conv = HypergraphConv(in_channels,
                          out_channels,
                          use_attention=True,
                          heads=2,
                          add_self_loops=False)
    conv_harness(conv,
                 batch=(x, hyperedge_index, hyperedge_weight, hyperedge_attr))


def test_hypergraph_conv_with_more_edges_than_nodes(request):
    pytest.skip(
        f'{request.node.nodeid}: AFS-144: Could not run '
        'aten::_local_scalar_dense with arguments from the Meta backend.')

    in_channels, out_channels = (16, 32)
    hyperedge_index = torch.tensor([[0, 0, 1, 1, 2, 3, 3, 3, 2, 1, 2],
                                    [0, 1, 2, 1, 2, 1, 0, 3, 3, 4, 4]])
    hyperedge_weight = torch.tensor([1.0, 0.5, 0.8, 0.2, 0.7])
    num_nodes = hyperedge_index[0].max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = HypergraphConv(in_channels, out_channels)

    conv_harness(conv, batch=(x, hyperedge_index))
    conv_harness(conv, batch=(x, hyperedge_index, hyperedge_weight))
