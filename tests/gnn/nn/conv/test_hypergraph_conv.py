# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import torch
from torch_geometric.nn import HypergraphConv

from conv_utils import conv_harness


def test_hypergraph_conv_with_more_nodes_than_edges():
    torch.manual_seed(42)
    in_channels, out_channels = (16, 32)
    hyperedge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [0, 1, 0, 1, 0, 1]])
    hyperedge_weight = torch.tensor([1.0, 0.5])
    num_nodes = hyperedge_index[0].max().item() + 1
    num_edges = hyperedge_index[1].max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    hyperedge_attr = torch.randn((num_edges, in_channels))

    conv = HypergraphConv(in_channels, out_channels, add_self_loops=False)

    conv_harness(conv, batch=(x, hyperedge_index, None, None, num_edges))

    conv = HypergraphConv(in_channels,
                          out_channels,
                          use_attention=True,
                          heads=2,
                          add_self_loops=False)

    conv_harness(conv,
                 batch=(x, hyperedge_index, hyperedge_weight, hyperedge_attr,
                        num_edges))


def test_hypergraph_conv_with_more_edges_than_nodes():
    torch.manual_seed(42)
    in_channels, out_channels = (16, 32)
    hyperedge_index = torch.tensor([[0, 0, 1, 1, 2, 3, 3, 3, 2, 1, 2],
                                    [0, 1, 2, 1, 2, 1, 0, 3, 3, 4, 4]])
    hyperedge_weight = torch.tensor([1.0, 0.5, 0.8, 0.2, 0.7])
    num_nodes = hyperedge_index[0].max().item() + 1
    num_edges = hyperedge_index[1].max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = HypergraphConv(in_channels, out_channels)

    conv_harness(conv, batch=(x, hyperedge_index, None, None, num_edges))
    conv_harness(conv,
                 batch=(x, hyperedge_index, hyperedge_weight, None, num_edges))
