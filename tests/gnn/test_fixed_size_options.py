# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest

import torch_geometric as pyg
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader, NeighborLoader

from poptorch_geometric.fixed_size_options import FixedSizeOptions


@pytest.mark.parametrize('dataset,expected_result',
                         [('fake_large_dataset',
                           FixedSizeOptions(
                               num_nodes=109,
                               num_edges=1099,
                               num_graphs=10,
                           )),
                          ('fake_hetero_dataset',
                           FixedSizeOptions(
                               num_nodes={
                                   "v0": 559,
                                   "v1": 559
                               },
                               num_edges={
                                   ("v0", "e0", "v0"): 5212,
                                   ("v1", "e0", "v1"): 5176,
                                   ("v0", "e0", "v1"): 5239,
                                   ("v1", "e0", "v0"): 5149,
                                   ("v0", "e1", "v1"): 5176,
                               },
                               num_graphs=10,
                           ))])
def test_fixed_size_options_from_dataset(dataset, expected_result, request):
    dataset = request.getfixturevalue(dataset)

    batch_size = 10
    fixed_size_options = FixedSizeOptions.from_dataset(dataset, batch_size)

    assert fixed_size_options.num_nodes == expected_result.num_nodes
    assert fixed_size_options.num_edges == expected_result.num_edges
    assert fixed_size_options.num_graphs == expected_result.num_graphs

    # With sample limit
    fixed_size_options = FixedSizeOptions.from_dataset(dataset,
                                                       batch_size,
                                                       sample_limit=10000)

    assert fixed_size_options.num_nodes == expected_result.num_nodes
    assert fixed_size_options.num_edges == expected_result.num_edges
    assert fixed_size_options.num_graphs == expected_result.num_graphs


@pytest.mark.parametrize('dataset,expected_result',
                         [('fake_large_dataset',
                           FixedSizeOptions(
                               num_nodes=116,
                               num_edges=1015,
                               num_graphs=11,
                           )),
                          ('fake_hetero_dataset',
                           FixedSizeOptions(
                               num_nodes={
                                   "v0": 543,
                                   "v1": 523
                               },
                               num_edges={
                                   ("v0", "e0", "v0"): 4950,
                                   ("v1", "e0", "v1"): 4766,
                                   ("v0", "e0", "v1"): 4897,
                                   ("v1", "e0", "v0"): 4667,
                                   ("v0", "e1", "v1"): 4914,
                               },
                               num_graphs=11,
                           ))])
def test_fixed_size_options_from_dataloader(dataset, expected_result, request):
    dataset = request.getfixturevalue(dataset)

    batch_size = 10
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    fixed_size_options = FixedSizeOptions.from_loader(dataloader)

    assert fixed_size_options.num_nodes == expected_result.num_nodes
    assert fixed_size_options.num_edges == expected_result.num_edges
    assert fixed_size_options.num_graphs == expected_result.num_graphs

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # With sample limit
    fixed_size_options = FixedSizeOptions.from_loader(dataloader,
                                                      sample_limit=1000)

    assert fixed_size_options.num_nodes == expected_result.num_nodes
    assert fixed_size_options.num_edges == expected_result.num_edges
    assert fixed_size_options.num_graphs == expected_result.num_graphs


@pytest.mark.parametrize('dataset,expected_result',
                         [('fake_node_task_dataset',
                           FixedSizeOptions(
                               num_nodes=13,
                               num_edges=61,
                               num_graphs=2,
                           )),
                          ('fake_node_task_hetero_dataset',
                           FixedSizeOptions(
                               num_nodes={
                                   "v0": 62,
                                   "v1": 43
                               },
                               num_edges={
                                   ("v0", "e0", "v0"): 146,
                                   ("v1", "e0", "v1"): 115,
                                   ("v0", "e0", "v1"): 116,
                                   ("v1", "e0", "v0"): 139,
                                   ("v0", "e1", "v1"): 116,
                               },
                               num_graphs=2,
                           ))])
def test_fixed_size_options_from_sample_dataloader(dataset, expected_result,
                                                   request):
    dataset = request.getfixturevalue(dataset)
    is_HeteroData = isinstance(dataset[0], HeteroData)

    pyg.seed_everything(42)
    dataloader = NeighborLoader(dataset[0], [5, 5],
                                batch_size=5,
                                shuffle=False,
                                input_nodes=("v0",
                                             None) if is_HeteroData else None)

    fixed_size_options = FixedSizeOptions.from_loader(dataloader)

    assert fixed_size_options.num_nodes == expected_result.num_nodes
    assert fixed_size_options.num_edges == expected_result.num_edges
    assert fixed_size_options.num_graphs == expected_result.num_graphs

    pyg.seed_everything(42)
    dataloader = NeighborLoader(dataset[0], [5, 5],
                                batch_size=5,
                                shuffle=False,
                                input_nodes=("v0",
                                             None) if is_HeteroData else None)

    # With sample limit
    fixed_size_options = FixedSizeOptions.from_loader(dataloader,
                                                      sample_limit=1000)

    assert fixed_size_options.num_nodes == expected_result.num_nodes
    assert fixed_size_options.num_edges == expected_result.num_edges
    assert fixed_size_options.num_graphs == expected_result.num_graphs


def test_fixed_size_options_to_hetero(request):
    dataset = request.getfixturevalue("fake_hetero_dataset")

    batch_size = 10
    num_nodes = 20
    num_edges = 40
    fixed_size_options = FixedSizeOptions(num_nodes=num_nodes,
                                          num_edges=num_edges,
                                          num_graphs=batch_size)
    fixed_size_options.to_hetero(dataset[0].node_types, dataset[0].edge_types)

    assert all(n == num_nodes for n in fixed_size_options.num_nodes.values())
    assert all(n == num_edges for n in fixed_size_options.num_edges.values())
    assert fixed_size_options.num_graphs == batch_size
