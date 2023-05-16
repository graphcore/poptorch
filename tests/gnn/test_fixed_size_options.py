# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest

from torch_geometric.loader import DataLoader, NeighborLoader

from poptorch_geometric.fixed_size_options import FixedSizeOptions


@pytest.mark.parametrize('dataset', ['fake_large_dataset'])
def test_fixed_size_options_from_dataset(dataset, request):
    dataset = request.getfixturevalue(dataset)

    batch_size = 10
    fixed_size_options = FixedSizeOptions.from_dataset(dataset, batch_size)

    assert fixed_size_options.num_nodes == 109
    assert fixed_size_options.num_edges == 1099
    assert fixed_size_options.num_graphs == 10

    # With sample limit
    fixed_size_options = FixedSizeOptions.from_dataset(dataset,
                                                       batch_size,
                                                       sample_limit=100)

    assert fixed_size_options.num_nodes == 109
    assert fixed_size_options.num_edges == 1099
    assert fixed_size_options.num_graphs == 10


@pytest.mark.parametrize('dataset', ['fake_large_dataset'])
def test_fixed_size_options_from_dataloader(dataset, request):
    dataset = request.getfixturevalue(dataset)

    batch_size = 10
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    fixed_size_options = FixedSizeOptions.from_loader(dataloader)

    assert fixed_size_options.num_nodes == 116
    assert fixed_size_options.num_edges == 1015
    assert fixed_size_options.num_graphs == 11

    # With sample limit
    fixed_size_options = FixedSizeOptions.from_loader(dataloader,
                                                      sample_limit=100)

    assert fixed_size_options.num_nodes == 116
    assert fixed_size_options.num_edges == 1015
    assert fixed_size_options.num_graphs == 11


@pytest.mark.parametrize('dataset', ['fake_node_task_dataset'])
def test_fixed_size_options_from_sample_dataloader(dataset, request):
    dataset = request.getfixturevalue(dataset)

    dataloader = NeighborLoader(dataset[0], [5, 5],
                                batch_size=5,
                                shuffle=False)

    fixed_size_options = FixedSizeOptions.from_loader(dataloader)

    assert fixed_size_options.num_nodes == 13
    assert fixed_size_options.num_edges == 61
    assert fixed_size_options.num_graphs == 2

    # With sample limit
    fixed_size_options = FixedSizeOptions.from_loader(dataloader,
                                                      sample_limit=100)

    assert fixed_size_options.num_nodes == 13
    assert fixed_size_options.num_edges == 61
    assert fixed_size_options.num_graphs == 2
