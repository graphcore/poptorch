# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
import torch_geometric as pyg
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

from poptorch_geometric.collate import FixedSizeCollater

# pylint: disable=protected-access


@pytest.mark.parametrize("num_graphs", [10, None])
@pytest.mark.parametrize("num_edges", [300, None])
@pytest.mark.parametrize("set_pad_values", [True, False])
def test_batch_masks(num_graphs, num_edges, set_pad_values):
    num_real_graphs = 8
    avg_num_nodes = 10
    num_channels = 8
    dataset = pyg.datasets.FakeDataset(num_graphs=16,
                                       avg_num_nodes=avg_num_nodes,
                                       avg_degree=2,
                                       num_channels=num_channels,
                                       edge_dim=2,
                                       task="graph")

    node_pad_value = 22.0 if set_pad_values else 0.0
    edge_pad_value = 34.0 if set_pad_values else 0.0
    graph_pad_value = 55.0 if set_pad_values else 0.0

    num_batch_nodes = 100
    num_batch_edges = num_batch_nodes * (num_batch_nodes - 1) \
         if num_edges is None else num_edges
    num_batch_graphs = num_real_graphs + 1 \
         if num_graphs is None else num_graphs

    fixed_size_collater = None
    if set_pad_values:
        fixed_size_collater = FixedSizeCollater(
            num_nodes=num_batch_nodes,
            num_edges=num_edges,
            num_graphs=num_graphs,
            add_masks_to_batch=True,
            node_pad_value=node_pad_value,
            edge_pad_value=edge_pad_value,
            graph_pad_value=graph_pad_value)
    else:
        fixed_size_collater = FixedSizeCollater(num_nodes=num_batch_nodes,
                                                num_edges=num_edges,
                                                num_graphs=num_graphs,
                                                add_masks_to_batch=True)

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


def test_prune_nodes_single_input(molecule):
    assert molecule.num_nodes == 29
    expected_num_nodes = 10
    fixed_size_collater = FixedSizeCollater(num_nodes=expected_num_nodes)
    result = fixed_size_collater._prune_nodes([molecule])
    assert len(result) == 1
    assert result[0].num_nodes == expected_num_nodes
    assert result[0].x.shape[0] == expected_num_nodes
    assert result[0].pos.shape[0] == expected_num_nodes


def test_prune_nodes_multiple_inputs(molecule):
    assert molecule.num_nodes == 29
    num_inputs = 4
    input = [molecule] * num_inputs
    expected_num_nodes = 80
    fixed_size_collater = FixedSizeCollater(num_nodes=expected_num_nodes)
    result = fixed_size_collater._prune_nodes(input)
    assert len(result) == num_inputs

    num_nodes = 0
    for data in result:
        num_nodes += data.num_nodes
        assert num_nodes > 0

    assert num_nodes == expected_num_nodes


def test_prune_nodes_multiple_inputs_minimal_num_node(molecule):
    assert molecule.num_nodes == 29
    num_inputs = 3
    input = [molecule] * num_inputs
    expected_num_nodes = num_inputs
    fixed_size_collater = FixedSizeCollater(num_nodes=expected_num_nodes)
    result = fixed_size_collater._prune_nodes(input)
    assert len(result) == num_inputs

    num_nodes = 0
    for data in result:
        num_nodes += data.num_nodes
        assert data.num_nodes > 0

    assert num_nodes == expected_num_nodes


def test_prune_edges_single_input(molecule):
    assert molecule.num_nodes == 29
    assert molecule.num_edges == 56

    expected_num_nodes = 29
    expected_num_edges = 40

    fixed_size_collator = FixedSizeCollater(num_nodes=expected_num_nodes,
                                            num_edges=expected_num_edges)

    result = fixed_size_collator._prune_edges([molecule])
    assert len(result) == 1
    assert result[0].num_nodes == expected_num_nodes
    assert result[0].x.shape[0] == expected_num_nodes
    assert result[0].pos.shape[0] == expected_num_nodes

    assert result[0].num_edges == expected_num_edges
    assert result[0].edge_attr.shape[0] == expected_num_edges
    assert result[0].edge_index.shape[1] == expected_num_edges


def test_prune_edges_multiple_inputs(molecule):
    assert molecule.num_nodes == 29
    assert molecule.num_edges == 56

    num_inputs = 4
    input = [molecule] * num_inputs
    expected_num_nodes = molecule.num_nodes * num_inputs
    expected_num_edges = 80

    fixed_size_collator = FixedSizeCollater(num_nodes=expected_num_nodes,
                                            num_edges=expected_num_edges)

    result = fixed_size_collator._prune_edges(input)
    assert len(result) == num_inputs

    num_nodes = 0
    num_edges = 0
    for data in result:
        assert data.num_nodes > 0
        num_nodes += data.num_nodes

        assert data.num_edges > 0
        num_edges += data.num_edges

    assert num_nodes == expected_num_nodes


def test_prune_nodes_multiple_inputs_minimal_num_edges(molecule):
    assert molecule.num_nodes == 29
    assert molecule.num_edges == 56

    num_inputs = 3
    input = [molecule] * num_inputs
    expected_num_nodes = molecule.num_nodes * num_inputs
    expected_num_edges = num_inputs

    fixed_size_collator = FixedSizeCollater(num_nodes=expected_num_nodes,
                                            num_edges=expected_num_edges)

    result = fixed_size_collator._prune_edges(input)
    assert len(result) == num_inputs

    num_nodes = 0
    num_edges = 0
    for data in result:
        assert data.num_nodes > 0
        num_nodes += data.num_nodes
        num_edges += data.num_edges

    assert num_nodes == expected_num_nodes


def test_prune_nodes_multiple_inputs_should_throw_exception(molecule):
    assert molecule.num_nodes == 29
    num_inputs = 3
    input = [molecule] * num_inputs
    expected_num_nodes = num_inputs - 1
    fixed_size_collater = FixedSizeCollater(num_nodes=expected_num_nodes)

    with pytest.raises(
            RuntimeError,
            match='Too many nodes to trim. Batch has 3 graphs with 87 total '
            'nodes. Requested to trim it to 85 nodes, which would result '
            'in empty graphs.'):
        fixed_size_collater._prune_nodes(input)


def test_prune_nodes_fixed_size_collater():
    avg_num_nodes = 30
    num_channels = 16
    batch_size = 10
    dataset = pyg.datasets.FakeDataset(num_graphs=99,
                                       avg_num_nodes=avg_num_nodes,
                                       avg_degree=5,
                                       num_channels=num_channels,
                                       edge_dim=8)

    expected_num_nodes = 80
    fixed_size_collater = FixedSizeCollater(num_nodes=expected_num_nodes,
                                            trim_nodes=True)

    batch_sampler = BatchSampler(RandomSampler(dataset),
                                 batch_size,
                                 drop_last=False)
    for sample in batch_sampler:
        result = fixed_size_collater([dataset[id] for id in sample])
        assert result.num_nodes == expected_num_nodes
        assert result.batch.shape[0] == expected_num_nodes
        assert result.x.shape[0] == expected_num_nodes


def test_prune_edges_fixed_size_collator():
    avg_num_nodes = 30
    num_channels = 16
    batch_size = 10
    dataset = pyg.datasets.FakeDataset(num_graphs=99,
                                       avg_num_nodes=avg_num_nodes,
                                       avg_degree=5,
                                       num_channels=num_channels,
                                       edge_dim=8)

    expected_num_nodes = avg_num_nodes * (batch_size * 2)
    expected_num_edges = 30

    fixed_size_collator = FixedSizeCollater(num_nodes=expected_num_nodes,
                                            num_edges=expected_num_edges,
                                            trim_edges=True)

    batch_sampler = BatchSampler(RandomSampler(dataset),
                                 batch_size,
                                 drop_last=False)
    for sample in batch_sampler:
        result = fixed_size_collator([dataset[id] for id in sample])

        assert result.num_nodes == expected_num_nodes
        assert result.batch.shape[0] == expected_num_nodes
        assert result.x.shape[0] == expected_num_nodes

        assert result.num_edges == expected_num_edges
        assert result.edge_attr.shape[0] == expected_num_edges
        assert result.edge_index.shape[1] == expected_num_edges


def test_prune_data_fixed_size_collator():
    avg_num_nodes = 30
    num_channels = 16
    batch_size = 10
    dataset = pyg.datasets.FakeDataset(num_graphs=99,
                                       avg_num_nodes=avg_num_nodes,
                                       avg_degree=5,
                                       num_channels=num_channels,
                                       edge_dim=8)

    expected_num_nodes = 100
    expected_num_edges = 30

    fixed_size_collator = FixedSizeCollater(num_nodes=expected_num_nodes,
                                            num_edges=expected_num_edges,
                                            trim_nodes=True,
                                            trim_edges=True)

    batch_sampler = BatchSampler(RandomSampler(dataset),
                                 batch_size,
                                 drop_last=False)
    for sample in batch_sampler:
        result = fixed_size_collator([dataset[id] for id in sample])

        assert result.num_nodes == expected_num_nodes
        assert result.batch.shape[0] == expected_num_nodes
        assert result.x.shape[0] == expected_num_nodes

        assert result.num_edges == expected_num_edges
        assert result.edge_attr.shape[0] == expected_num_edges
        assert result.edge_index.shape[1] == expected_num_edges


def test_valid_args_fixed_size_collater(molecule):
    num_inputs = 3
    expected_num_nodes = molecule.num_nodes * num_inputs

    fixed_size_collater = FixedSizeCollater(num_nodes=expected_num_nodes)
    input_list = [molecule] * num_inputs
    fixed_size_collater(input_list)

    with pytest.raises(TypeError, match='Expected list, got tuple.'):
        fixed_size_collater(tuple(input_list))


def test_fixed_size_collater_should_include_non_tensor_keys_in_pad_graph(
        molecule):
    expected_num_nodes = molecule.num_nodes * 3
    molecule['scalar_key'] = 2

    fixed_size_collater = FixedSizeCollater(num_nodes=expected_num_nodes)
    input_list = [molecule]
    result = fixed_size_collater(input_list)
    assert result.name == ['gdb_57518', 'gdb_57518']
    assert torch.equal(result.scalar_key, torch.Tensor([2, 2]))


def test_fixed_size_collater_should_assign_default_pad_values(molecule):
    expected_num_nodes = molecule.num_nodes * 3
    molecule['scalar_key'] = 2
    pad_graph_defaults = {'name': 'pad_graph', 'scalar_key': 3}
    input_list = [molecule]

    fixed_size_collater = FixedSizeCollater(
        num_nodes=expected_num_nodes, pad_graph_defaults=pad_graph_defaults)
    result = fixed_size_collater(input_list)
    assert result.name == ['gdb_57518', 'pad_graph']
    assert torch.equal(result.scalar_key, torch.Tensor([2, 3]))
