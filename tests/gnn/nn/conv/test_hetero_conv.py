# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch

from torch_geometric.data import HeteroData
from torch_geometric.nn import (GATConv, GCNConv, HeteroConv, Linear,
                                MessagePassing, SAGEConv, to_hetero)
import torch_geometric.transforms as T

from conv_utils import hetero_conv_harness


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


def get_dummy_data():
    data = HeteroData()
    data['paper'].x = torch.randn(50, 32)
    data['author'].x = torch.randn(30, 64)
    data['paper', 'paper'].edge_index = get_edge_index(50, 50, 200)
    data['paper', 'author'].edge_index = get_edge_index(50, 30, 100)
    data['paper', 'author'].edge_attr = torch.randn(100, 3)
    data['author', 'paper'].edge_index = get_edge_index(30, 50, 100)
    data['paper', 'paper'].edge_weight = torch.rand(200)
    data['author', 'author'].edge_index = get_edge_index(30, 30, 100)
    return data


@pytest.mark.parametrize('aggr', ['sum', 'mean', 'min', 'max', 'cat', None])
def test_hetero_conv(aggr):
    data = get_dummy_data()

    conv = HeteroConv(
        {
            ('paper', 'to', 'paper'):
            GCNConv(-1, 64, add_self_loops=False),
            ('author', 'to', 'paper'):
            SAGEConv((-1, -1), 64, add_self_loops=False),
            ('paper', 'to', 'author'):
            GATConv((-1, -1), 64, edge_dim=3, add_self_loops=False),
        },
        aggr=aggr)

    _ = conv(data.x_dict,
             data.edge_index_dict,
             data.edge_attr_dict,
             edge_weight_dict=data.edge_weight_dict)

    forward_args = ('x_dict', 'edge_index_dict', 'edge_attr_dict',
                    'edge_weight_dict')
    hetero_conv_harness(conv, data, 'author', forward_args=forward_args)


@pytest.mark.parametrize('aggr', ['sum', 'mean', 'min', 'max', 'cat'])
@pytest.mark.parametrize('num_layers', [2, 5])
def test_hetero_conv_multiple_layers(aggr, num_layers):
    data = get_dummy_data()

    class MultiLayerHeteroConv(torch.nn.Module):
        def __init__(self, num_layers):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(
                    HeteroConv(
                        {
                            ('paper', 'to', 'paper'):
                            GCNConv(-1, 64, add_self_loops=False),
                            ('author', 'to', 'paper'):
                            SAGEConv((-1, -1), 64, add_self_loops=False),
                            ('paper', 'to', 'author'):
                            GATConv(
                                (-1, -1), 64, edge_dim=3,
                                add_self_loops=False),
                        },
                        aggr=aggr))

        def forward(self, x_dict, edge_index_dict, *args, **kwargs):
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict, *args, **kwargs)
                x_dict = {key: x.relu() for key, x in x_dict.items()}
            return x_dict

    conv = MultiLayerHeteroConv(num_layers)

    _ = conv(data.x_dict,
             data.edge_index_dict,
             data.edge_attr_dict,
             edge_weight_dict=data.edge_weight_dict)

    forward_args = ('x_dict', 'edge_index_dict', 'edge_attr_dict',
                    'edge_weight_dict')
    hetero_conv_harness(conv, data, 'author', forward_args=forward_args)


@pytest.mark.parametrize('aggr', ['sum', 'mean', 'min', 'max', 'cat'])
@pytest.mark.parametrize('num_layers', [2, 5])
def test_hetero_conv_multiple_layers_with_data_transforms(aggr, num_layers):
    data = get_dummy_data()
    data = T.ToUndirected()(data)
    data = T.AddSelfLoops()(data)
    data = T.NormalizeFeatures()(data)

    class MultiLayerHeteroConv(torch.nn.Module):
        def __init__(self, num_layers):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(
                    HeteroConv(
                        {
                            ('paper', 'to', 'paper'):
                            GCNConv(-1, 64, add_self_loops=False),
                            ('author', 'to', 'author'):
                            GCNConv(-1, 64, add_self_loops=False),
                            ('author', 'to', 'paper'):
                            SAGEConv((-1, -1), 64, add_self_loops=False),
                            ('paper', 'to', 'author'):
                            GATConv(
                                (-1, -1), 64, edge_dim=3,
                                add_self_loops=False),
                            ('paper', 'rev_to', 'author'):
                            SAGEConv((-1, -1), 64, add_self_loops=False),
                            ('author', 'rev_to', 'paper'):
                            GATConv(
                                (-1, -1), 64, edge_dim=3,
                                add_self_loops=False),
                        },
                        aggr=aggr))

        def forward(self, x_dict, edge_index_dict, *args, **kwargs):
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict, *args, **kwargs)
                x_dict = {key: x.relu() for key, x in x_dict.items()}
            return x_dict

    conv = MultiLayerHeteroConv(num_layers)

    _ = conv(data.x_dict,
             data.edge_index_dict,
             data.edge_attr_dict,
             edge_weight_dict=data.edge_weight_dict)

    forward_args = ('x_dict', 'edge_index_dict', 'edge_attr_dict',
                    'edge_weight_dict')
    hetero_conv_harness(conv, data, 'author', forward_args=forward_args)


# pylint: disable=abstract-method
# pylint: disable=arguments-differ
class CustomConv(MessagePassing):
    def __init__(self, out_channels):
        super().__init__(aggr='add')
        self.lin = Linear(-1, out_channels)

    def forward(self, x, edge_index, y, z):
        return self.propagate(edge_index, x=x, y=y, z=z)

    def message(self, x_j, y_j, z_j):
        return self.lin(torch.cat([x_j, y_j, z_j], dim=-1))


def test_hetero_conv_with_custom_conv():
    data = HeteroData()
    data['paper'].x = torch.randn(50, 32)
    data['paper'].y = torch.randn(50, 3)
    data['paper'].z = torch.randn(50, 3)
    data['author'].x = torch.randn(30, 64)
    data['author'].y = torch.randn(30, 3)
    data['author'].z = torch.randn(30, 3)
    data['paper', 'paper'].edge_index = get_edge_index(50, 50, 200)
    data['paper', 'author'].edge_index = get_edge_index(50, 30, 100)
    data['author', 'paper'].edge_index = get_edge_index(30, 50, 100)

    conv = HeteroConv({key: CustomConv(64) for key in data.edge_types})

    _ = conv(data.x_dict, data.edge_index_dict, data.y_dict, data.z_dict)

    forward_args = ('x_dict', 'edge_index_dict', 'y_dict', 'z_dict')
    hetero_conv_harness(conv, data, 'author', forward_args=forward_args)


@pytest.mark.parametrize('aggr', ['sum', 'mean', 'min', 'max'])
def test_to_hetero_transformation_basic(aggr):
    data = get_dummy_data()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), 64)
            self.conv2 = SAGEConv((-1, -1), 64)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x

    model = Model()
    model = to_hetero(model, data.metadata(), aggr=aggr)

    _ = model(data.x_dict, data.edge_index_dict)

    forward_args = ('x_dict', 'edge_index_dict')
    hetero_conv_harness(model, data, 'author', forward_args=forward_args)


@pytest.mark.parametrize('aggr', ['sum', 'mean', 'min', 'max'])
def test_to_hetero_transformation_skip_connections(aggr):
    data = get_dummy_data()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), 64)
            self.lin1 = Linear(-1, 64)
            self.conv2 = SAGEConv((-1, -1), 64)
            self.lin2 = Linear(-1, 64)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index) + self.lin1(x)
            x = x.relu()
            x = self.conv2(x, edge_index) + self.lin2(x)
            return x

    model = Model()
    model = to_hetero(model, data.metadata(), aggr=aggr)

    _ = model(data.x_dict, data.edge_index_dict)

    forward_args = ('x_dict', 'edge_index_dict')
    hetero_conv_harness(model, data, 'author', forward_args=forward_args)
