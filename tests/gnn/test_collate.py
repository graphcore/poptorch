# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest

import torch

import torch_geometric as pyg
from torch_geometric.data import Data, HeteroData
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

from utils import is_data
from poptorch_geometric.collate import FixedSizeCollater
from poptorch_geometric.fixed_size_options import FixedSizeOptions

# pylint: disable=protected-access


@pytest.fixture(params=[Data, HeteroData])
def _get_test_data(request, molecule, fake_hetero_data):
    if is_data(request.param):
        dataset = molecule
        assert dataset.num_nodes == 29
        assert dataset.num_edges == 56
    else:
        dataset = fake_hetero_data
        dataset['name'] = 'gdb_57518'
        assert dataset.num_nodes == 103
        assert dataset.num_edges == 2391
    return request.param, dataset


@pytest.mark.parametrize('num_graphs,num_real_graphs', [(10, 8), (2, 1)])
@pytest.mark.parametrize('num_edges', [300, None])
@pytest.mark.parametrize('set_pad_values', [True, False])
def test_batch_masks(num_graphs, num_real_graphs, num_edges, set_pad_values):
    avg_num_nodes = 10
    num_channels = 8

    dataset = pyg.datasets.FakeDataset(num_graphs=16,
                                       avg_num_nodes=avg_num_nodes,
                                       avg_degree=2,
                                       num_channels=num_channels,
                                       edge_dim=2,
                                       task='graph')

    node_pad_value = 22.0 if set_pad_values else 0.0
    edge_pad_value = 34.0 if set_pad_values else 0.0
    graph_pad_value = 55.0 if set_pad_values else 0.0

    num_batch_nodes = 100
    num_batch_edges = num_batch_nodes * (num_batch_nodes - 1) \
         if num_edges is None else num_edges
    num_batch_graphs = num_graphs

    fixed_size_options = None
    fixed_size_collater = None
    if set_pad_values:
        fixed_size_options = FixedSizeOptions(num_nodes=num_batch_nodes,
                                              num_edges=num_edges,
                                              num_graphs=num_graphs,
                                              node_pad_value=node_pad_value,
                                              edge_pad_value=edge_pad_value,
                                              graph_pad_value=graph_pad_value)
        fixed_size_collater = FixedSizeCollater(
            fixed_size_options=fixed_size_options, add_masks_to_batch=True)
    else:
        fixed_size_options = FixedSizeOptions(
            num_nodes=num_batch_nodes,
            num_edges=num_edges,
            num_graphs=num_graphs,
        )
        fixed_size_collater = FixedSizeCollater(
            fixed_size_options=fixed_size_options, add_masks_to_batch=True)

    batch_sampler = BatchSampler(SequentialSampler(dataset),
                                 num_real_graphs,
                                 drop_last=False)

    for i, sample in enumerate(batch_sampler):
        num_real_nodes = sum(dataset[id].num_nodes for id in sample)
        num_real_edges = sum(dataset[id].num_edges for id in sample)
        result = fixed_size_collater([dataset[id] for id in sample])

        # Check graph values
        assert len(result.graphs_mask) == num_batch_graphs
        assert int(result.graphs_mask.sum()) == num_real_graphs

        for j, mask in enumerate(result.graphs_mask):
            if mask.item() is True:
                assert dataset[i * num_real_graphs + j].y[0] == result.y[j]
            else:
                assert result.y[j] == graph_pad_value

        # Check nodes values
        assert len(result.nodes_mask) == num_batch_nodes
        assert int(result.nodes_mask.sum()) == num_real_nodes

        begin = 0
        end = 0
        for id in sample:
            end += dataset[id].num_nodes
            assert torch.all(result.nodes_mask[begin:end])
            assert torch.equal(result.x[begin:end], dataset[id].x)
            begin += dataset[id].num_nodes

        assert not torch.any(result.nodes_mask[begin:])
        for node_features in result.x[begin:]:
            for feature in node_features:
                assert feature == node_pad_value

        # Check edges values
        assert len(result.edges_mask) == num_batch_edges
        assert int(result.edges_mask.sum()) == num_real_edges

        begin = 0
        end = 0
        for id in sample:
            end += dataset[id].num_edges
            assert torch.all(result.edges_mask[begin:end])
            assert torch.equal(result.edge_attr[begin:end],
                               dataset[id].edge_attr)
            begin += dataset[id].num_edges

        assert not torch.any(result.edges_mask[begin:])
        for edge_features in result.edge_attr[begin:]:
            for feature in edge_features:
                assert feature == edge_pad_value


@pytest.mark.parametrize('num_graphs,num_real_graphs', [(6, 2), (4, 2),
                                                        (2, 1)])
@pytest.mark.parametrize('num_edges', [1200, None])
@pytest.mark.parametrize('set_pad_values', [True, False])
def test_batch_masks_heterodata(num_graphs, num_real_graphs, num_edges,
                                set_pad_values, fake_hetero_dataset):

    dataset = fake_hetero_dataset
    num_node_types = 2
    num_edge_types = 5

    node_pad_value = 22.0 if set_pad_values else 0.0
    edge_pad_value = 34.0 if set_pad_values else 0.0
    graph_pad_value = 55.0 if set_pad_values else 0.0

    num_batch_nodes = 150

    fixed_size_options = None
    fixed_size_collater = None
    if set_pad_values:
        fixed_size_options = FixedSizeOptions(num_nodes=num_batch_nodes,
                                              num_edges=num_edges,
                                              num_graphs=num_graphs,
                                              node_pad_value=node_pad_value,
                                              edge_pad_value=edge_pad_value,
                                              graph_pad_value=graph_pad_value)
        fixed_size_collater = FixedSizeCollater(
            fixed_size_options=fixed_size_options, add_masks_to_batch=True)
    else:
        fixed_size_options = FixedSizeOptions(
            num_nodes=num_batch_nodes,
            num_edges=num_edges,
            num_graphs=num_graphs,
        )
        fixed_size_collater = FixedSizeCollater(
            fixed_size_options=fixed_size_options, add_masks_to_batch=True)

    num_batch_edges = (num_batch_nodes * (num_batch_nodes - 1) \
         if num_edges is None else num_edges) * num_edge_types
    num_batch_graphs = num_graphs
    num_batch_nodes *= num_node_types

    batch_sampler = BatchSampler(SequentialSampler(dataset),
                                 num_real_graphs,
                                 drop_last=False)

    for i, sample in enumerate(batch_sampler):
        num_real_nodes = sum(dataset[id].num_nodes for id in sample)
        num_real_edges = sum(dataset[id].num_edges for id in sample)
        result = fixed_size_collater([dataset[id] for id in sample])

        # Check graph values
        assert len(result.graphs_mask) == num_batch_graphs
        assert int(result.graphs_mask.sum()) == num_real_graphs

        for j, mask in enumerate(result.graphs_mask):
            if mask.item() is True:
                assert dataset[i * num_real_graphs + j].y[0] == result.y[j]
            else:
                assert result.y[j] == graph_pad_value

        # Check nodes values
        assert sum(node_type.nodes_mask.shape[0]
                   for node_type in result.node_stores) == num_batch_nodes

        for key in result.node_types:
            num_real_nodes = sum(
                dataset[id]._node_store_dict[key]['x'].shape[0]
                for id in sample)
            nodes_mask = result._node_store_dict[key]['nodes_mask']
            assert torch.all(nodes_mask[0:num_real_nodes])
            assert not torch.any(nodes_mask[num_real_nodes:])
            x = result._node_store_dict[key]['x']
            assert not torch.all(x[num_real_nodes:] - node_pad_value)

        # Check edges values
        assert sum(edge_type.edges_mask.shape[0]
                   for edge_type in result.edge_stores) == num_batch_edges

        for key in result.edge_types:
            num_real_edges = sum(
                dataset[id]._edge_store_dict[key]['edge_index'].shape[1]
                for id in sample)
            edges_mask = result._edge_store_dict[key]['edges_mask']
            assert torch.all(edges_mask[0:num_real_edges])
            assert not torch.any(edges_mask[num_real_edges:])


def test_prune_nodes_single_input(_get_test_data):
    type_, dataset = _get_test_data

    if is_data(type_):
        fixed_size_options = FixedSizeOptions(num_nodes=10, num_graphs=2)
    else:
        fixed_size_options = FixedSizeOptions(num_nodes=dict(v0=10, v1=5),
                                              num_graphs=2)
        fixed_size_options.to_hetero(dataset.node_types, dataset.edge_types)

    fixed_size_collater = FixedSizeCollater(fixed_size_options)
    result = fixed_size_collater._prune_nodes([dataset])
    assert len(result) == 1

    if is_data(type_):
        assert result[0].num_nodes == fixed_size_options.num_nodes
        assert result[0].x.shape[0] == fixed_size_options.num_nodes
        assert result[0].pos.shape[0] == fixed_size_options.num_nodes
    else:
        assert result[0].num_nodes == fixed_size_options.total_num_nodes
        for node_type, expected_val in fixed_size_options.num_nodes.items():
            assert result[0][node_type].num_nodes == expected_val
            assert result[0][node_type].x.shape[0] == expected_val


def test_prune_nodes_multiple_inputs(_get_test_data):
    type_, dataset = _get_test_data

    num_inputs = 4
    input = [dataset] * num_inputs
    if is_data(type_):
        fixed_size_options = FixedSizeOptions(num_nodes=80,
                                              num_graphs=num_inputs + 1)
    else:
        fixed_size_options = FixedSizeOptions(num_nodes=dict(v0=80, v1=40),
                                              num_graphs=num_inputs + 1)
        fixed_size_options.to_hetero(dataset.node_types, dataset.edge_types)

    fixed_size_collater = FixedSizeCollater(fixed_size_options)
    result = fixed_size_collater._prune_nodes(input)
    num_nodes = 0
    for data in result:
        num_nodes += data.num_nodes
        assert num_nodes > 0

    assert num_nodes == fixed_size_options.total_num_nodes


def test_prune_nodes_multiple_inputs_minimal_num_node(_get_test_data):
    type_, dataset = _get_test_data

    num_inputs = 3
    input = [dataset] * num_inputs

    if is_data(type_):
        fixed_size_options = FixedSizeOptions(num_nodes=3,
                                              num_graphs=num_inputs + 1)
    else:
        fixed_size_options = FixedSizeOptions(num_nodes=dict(v0=3, v1=3),
                                              num_graphs=num_inputs + 1)
        fixed_size_options.to_hetero(dataset.node_types, dataset.edge_types)

    fixed_size_collater = FixedSizeCollater(fixed_size_options)

    result = fixed_size_collater._prune_nodes(input)
    assert len(result) == num_inputs

    num_nodes = 0
    for data in result:
        num_nodes += data.num_nodes
        assert data.num_nodes > 0

    assert num_nodes == fixed_size_options.total_num_nodes


def test_prune_edges_single_input(_get_test_data):
    type_, dataset = _get_test_data

    if is_data(type_):
        fixed_size_options = FixedSizeOptions(num_nodes=dataset.num_nodes,
                                              num_edges=40)
    else:
        fixed_size_options = FixedSizeOptions(num_nodes=dict(
            v0=dataset["v0"].num_nodes, v1=dataset["v1"].num_nodes),
                                              num_edges={
                                                  ("v0", "e0", "v1"): 40,
                                                  ("v0", "e0", "v0"): 30,
                                                  ("v1", "e0", "v0"): 30,
                                                  ("v0", "e1", "v1"): 40,
                                                  ("v1", "e0", "v1"): 50,
                                              })

    fixed_size_collator = FixedSizeCollater(fixed_size_options)

    result = fixed_size_collator._prune_edges([dataset])

    assert len(result) == 1
    assert result[0].num_nodes == fixed_size_options.total_num_nodes
    assert result[0].num_edges == fixed_size_options.total_num_edges

    if is_data(type_):
        assert result[0].x.shape[0] == fixed_size_options.num_nodes
        assert result[0].pos.shape[0] == fixed_size_options.num_nodes
        assert result[0].edge_attr.shape[0] == fixed_size_options.num_edges
        assert result[0].edge_index.shape[1] == fixed_size_options.num_edges
    else:
        for edge_type, expected_num in fixed_size_options.num_edges.items():
            assert result[0][edge_type].edge_index.shape[1] == expected_num


def test_prune_edges_multiple_inputs(_get_test_data):
    type_, dataset = _get_test_data

    num_inputs = 4
    input = [dataset] * num_inputs

    if is_data(type_):
        fixed_size_options = FixedSizeOptions(num_nodes=dataset.num_nodes *
                                              num_inputs,
                                              num_edges=80,
                                              num_graphs=num_inputs + 1)
    else:
        fixed_size_options = FixedSizeOptions(num_nodes=dict(
            v0=dataset["v0"].num_nodes * num_inputs,
            v1=dataset["v1"].num_nodes * num_inputs),
                                              num_edges={
                                                  ("v0", "e0", "v1"): 80,
                                                  ("v0", "e0", "v0"): 120,
                                                  ("v1", "e0", "v0"): 90,
                                                  ("v0", "e1", "v1"): 100,
                                                  ("v1", "e0", "v1"): 80,
                                              })

    fixed_size_collator = FixedSizeCollater(fixed_size_options)

    result = fixed_size_collator._prune_edges(input)
    assert len(result) == num_inputs

    num_nodes = 0
    num_edges = 0
    for data in result:
        assert data.num_nodes > 0
        num_nodes += data.num_nodes

        assert data.num_edges > 0
        num_edges += data.num_edges

    assert num_nodes == fixed_size_options.total_num_nodes
    assert num_edges == fixed_size_options.total_num_edges


def test_prune_nodes_multiple_inputs_minimal_num_edges(_get_test_data):
    type_, dataset = _get_test_data

    num_inputs = 3
    input = [dataset] * num_inputs

    if is_data(type_):
        fixed_size_options = FixedSizeOptions(num_nodes=dataset.num_nodes *
                                              num_inputs,
                                              num_edges=80,
                                              num_graphs=num_inputs + 1)
    else:
        fixed_size_options = FixedSizeOptions(num_nodes=dict(
            v0=dataset["v0"].num_nodes * num_inputs,
            v1=dataset["v1"].num_nodes * num_inputs),
                                              num_edges={
                                                  ("v0", "e0", "v1"): 80,
                                                  ("v0", "e0", "v0"): 120,
                                                  ("v1", "e0", "v0"): 90,
                                                  ("v0", "e1", "v1"): 100,
                                                  ("v1", "e0", "v1"): 80,
                                              })

    fixed_size_collator = FixedSizeCollater(fixed_size_options)

    result = fixed_size_collator._prune_edges(input)
    assert len(result) == num_inputs

    num_nodes = 0
    num_edges = 0
    for data in result:
        assert data.num_nodes > 0
        num_nodes += data.num_nodes
        num_edges += data.num_edges

    assert num_nodes == fixed_size_options.total_num_nodes
    assert num_edges == fixed_size_options.total_num_edges


def test_prune_nodes_multiple_inputs_should_throw_exception(_get_test_data):
    type_, dataset = _get_test_data

    num_inputs = 3
    input = [dataset] * num_inputs
    expected_num_nodes = (num_inputs - 1)

    fixed_size_options = FixedSizeOptions(num_nodes=expected_num_nodes,
                                          num_graphs=num_inputs + 1)
    if not is_data(type_):
        fixed_size_options.to_hetero(dataset.node_types, dataset.edge_types)
    fixed_size_collater = FixedSizeCollater(fixed_size_options)

    with pytest.raises(RuntimeError):
        fixed_size_collater._prune_nodes(input)


@pytest.mark.parametrize('data_type,fixed_size_hetero', [(Data, False),
                                                         (HeteroData, False),
                                                         (HeteroData, True)])
def test_prune_nodes_fixed_size_collater(data_type, fixed_size_hetero,
                                         fake_hetero_dataset):
    batch_size = 10

    if is_data(data_type):
        avg_num_nodes = 30
        num_channels = 16
        dataset = pyg.datasets.FakeDataset(num_graphs=99,
                                           avg_num_nodes=avg_num_nodes,
                                           avg_degree=5,
                                           num_channels=num_channels,
                                           edge_dim=8)
    else:
        avg_num_nodes = 60
        dataset = fake_hetero_dataset

    if fixed_size_hetero:
        fixed_size_options = FixedSizeOptions(num_nodes=dict(v0=800, v1=800),
                                              num_graphs=batch_size + 1)
    else:
        fixed_size_options = FixedSizeOptions(num_nodes=800,
                                              num_graphs=batch_size + 1)

    fixed_size_collater = FixedSizeCollater(fixed_size_options,
                                            trim_nodes=True)
    batch_sampler = BatchSampler(RandomSampler(dataset),
                                 batch_size,
                                 drop_last=False)
    for sample in batch_sampler:
        result = fixed_size_collater([dataset[id] for id in sample])
        assert result.num_nodes == fixed_size_options.total_num_nodes
        assert result.num_edges == fixed_size_options.total_num_edges

        if is_data(data_type):
            assert result.batch.shape[0] == fixed_size_options.total_num_nodes
            assert result.x.shape[0] == fixed_size_options.total_num_nodes
            assert result.edge_attr.shape[
                0] == fixed_size_options.total_num_edges
            assert result.edge_index.shape[
                1] == fixed_size_options.total_num_edges
        else:
            for node_type, expected_val in fixed_size_options.num_nodes.items(
            ):
                assert result[node_type].num_nodes == expected_val
                assert result[node_type].x.shape[0] == expected_val
            for edge_type, expected_num in fixed_size_options.num_edges.items(
            ):
                assert result[edge_type].edge_index.shape[1] == expected_num


@pytest.mark.parametrize('data_type,fixed_size_hetero', [(Data, False),
                                                         (HeteroData, False),
                                                         (HeteroData, True)])
def test_prune_edges_fixed_size_collator(data_type, fixed_size_hetero,
                                         fake_hetero_dataset):
    batch_size = 10

    if is_data(data_type):
        avg_num_nodes = 30
        num_channels = 16
        dataset = pyg.datasets.FakeDataset(num_graphs=99,
                                           avg_num_nodes=avg_num_nodes,
                                           avg_degree=5,
                                           num_channels=num_channels,
                                           edge_dim=8)
    else:
        avg_num_nodes = 60
        dataset = fake_hetero_dataset

    if fixed_size_hetero:
        fixed_size_options = FixedSizeOptions(num_nodes=dict(
            v0=avg_num_nodes * (batch_size * 2),
            v1=avg_num_nodes * (batch_size * 2)),
                                              num_edges={
                                                  ("v0", "e0", "v1"): 80,
                                                  ("v0", "e0", "v0"): 120,
                                                  ("v1", "e0", "v0"): 90,
                                                  ("v0", "e1", "v1"): 100,
                                                  ("v1", "e0", "v1"): 80,
                                              },
                                              num_graphs=batch_size + 1)
    else:
        fixed_size_options = FixedSizeOptions(num_nodes=avg_num_nodes *
                                              (batch_size * 2),
                                              num_edges=30,
                                              num_graphs=batch_size + 1)

    fixed_size_collator = FixedSizeCollater(fixed_size_options,
                                            trim_edges=True)
    batch_sampler = BatchSampler(RandomSampler(dataset),
                                 batch_size,
                                 drop_last=False)
    for sample in batch_sampler:
        result = fixed_size_collator([dataset[id] for id in sample])

        assert result.num_nodes == fixed_size_options.total_num_nodes
        assert result.num_edges == fixed_size_options.total_num_edges

        if is_data(data_type):
            assert result.batch.shape[0] == fixed_size_options.total_num_nodes
            assert result.x.shape[0] == fixed_size_options.total_num_nodes
            assert result.edge_attr.shape[
                0] == fixed_size_options.total_num_edges
            assert result.edge_index.shape[
                1] == fixed_size_options.total_num_edges
        else:
            for node_type, expected_val in fixed_size_options.num_nodes.items(
            ):
                assert result[node_type].num_nodes == expected_val
                assert result[node_type].x.shape[0] == expected_val
            for edge_type, expected_num in fixed_size_options.num_edges.items(
            ):
                assert result[edge_type].edge_index.shape[1] == expected_num


@pytest.mark.parametrize('data_type,fixed_size_hetero', [(Data, False),
                                                         (HeteroData, False),
                                                         (HeteroData, True)])
def test_prune_data_fixed_size_collator(data_type, fixed_size_hetero,
                                        fake_hetero_dataset):
    batch_size = 10

    if is_data(data_type):
        avg_num_nodes = 30
        num_channels = 16
        dataset = pyg.datasets.FakeDataset(num_graphs=99,
                                           avg_num_nodes=avg_num_nodes,
                                           avg_degree=5,
                                           num_channels=num_channels,
                                           edge_dim=8)
    else:
        avg_num_nodes = 300
        dataset = fake_hetero_dataset

    if fixed_size_hetero:
        fixed_size_options = FixedSizeOptions(num_nodes=dict(v0=200, v1=100),
                                              num_edges={
                                                  ("v0", "e0", "v1"): 80,
                                                  ("v0", "e0", "v0"): 120,
                                                  ("v1", "e0", "v0"): 90,
                                                  ("v0", "e1", "v1"): 100,
                                                  ("v1", "e0", "v1"): 80,
                                              },
                                              num_graphs=batch_size + 1)
    else:
        fixed_size_options = FixedSizeOptions(num_nodes=200,
                                              num_edges=30,
                                              num_graphs=batch_size + 1)

    for data in dataset:
        if is_data(data_type):
            assert data.edge_index.shape[1] > 0
        else:
            for edge_store in data.edge_stores:
                assert edge_store['edge_index'].shape[1] > 0

    fixed_size_collator = FixedSizeCollater(fixed_size_options,
                                            trim_nodes=True,
                                            trim_edges=True)
    batch_sampler = BatchSampler(RandomSampler(dataset),
                                 batch_size,
                                 drop_last=False)
    for sample in batch_sampler:
        result = fixed_size_collator([dataset[id] for id in sample])

        assert result.num_nodes == fixed_size_options.total_num_nodes
        assert result.num_edges == fixed_size_options.total_num_edges

        if is_data(data_type):
            assert result.batch.shape[0] == fixed_size_options.total_num_nodes
            assert result.x.shape[0] == fixed_size_options.total_num_nodes
            assert result.edge_attr.shape[
                0] == fixed_size_options.total_num_edges
            assert result.edge_index.shape[
                1] == fixed_size_options.total_num_edges
        else:
            for node_type, expected_val in fixed_size_options.num_nodes.items(
            ):
                assert result[node_type].num_nodes == expected_val
                assert result[node_type].x.shape[0] == expected_val
            for edge_type, expected_num in fixed_size_options.num_edges.items(
            ):
                assert result[edge_type].edge_index.shape[1] == expected_num


def test_valid_args_fixed_size_collater(_get_test_data):
    _, dataset = _get_test_data

    num_inputs = 3
    expected_num_nodes = dataset.num_nodes * num_inputs

    fixed_size_options = FixedSizeOptions(num_nodes=expected_num_nodes,
                                          num_graphs=num_inputs + 1)
    fixed_size_collater = FixedSizeCollater(fixed_size_options)
    input_list = [dataset] * num_inputs
    fixed_size_collater(input_list)

    with pytest.raises(TypeError, match='Expected list, got tuple.'):
        fixed_size_collater(tuple(input_list))


def test_fixed_size_collater_should_include_non_tensor_keys_in_pad_graph(
        _get_test_data):
    _, dataset = _get_test_data

    dataset['scalar_key'] = 2
    expected_num_nodes = dataset.num_nodes * 3

    fixed_size_options = FixedSizeOptions(num_nodes=expected_num_nodes)
    fixed_size_collater = FixedSizeCollater(fixed_size_options)
    input_list = [dataset]
    result = fixed_size_collater(input_list)

    assert result.name == ['gdb_57518', 'gdb_57518']
    assert torch.equal(result.scalar_key, torch.Tensor([2, 2]))


def test_fixed_size_collater_should_assign_default_pad_values(_get_test_data):
    _, dataset = _get_test_data

    expected_num_nodes = dataset.num_nodes * 3
    dataset['scalar_key'] = 2
    pad_graph_defaults = {'name': 'pad_graph', 'scalar_key': 3}
    input_list = [dataset]

    fixed_size_options = FixedSizeOptions(
        num_nodes=expected_num_nodes, pad_graph_defaults=pad_graph_defaults)
    fixed_size_collater = FixedSizeCollater(fixed_size_options)
    result = fixed_size_collater(input_list)
    assert result.name == ['gdb_57518', 'pad_graph']
    assert torch.equal(result.scalar_key, torch.Tensor([2, 3]))


@pytest.mark.parametrize('num_nodes,num_edges,error_type',
                         [(10, 10000, 'nodes'), (10000, 10, 'edges')])
def test_fixed_size_collater_wrong_size_exceptions(_get_test_data, num_nodes,
                                                   num_edges, error_type):
    _, dataset = _get_test_data

    num_inputs = 4
    input = [dataset] * num_inputs
    fixed_size_options = FixedSizeOptions(num_nodes=num_nodes,
                                          num_edges=num_edges,
                                          num_graphs=num_inputs + 1)

    fixed_size_collater = FixedSizeCollater(fixed_size_options)

    error_contains = (
        r"The fixed sizes given don't allocate enough space for the"
        fr" number of .* {error_type}")

    with pytest.raises(RuntimeError, match=error_contains):
        # TODO: Be more specific about error
        fixed_size_collater(input)
