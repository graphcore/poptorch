# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import itertools
import math
from statistics import mean

import pytest
import torch
from utils import FakeDatasetEqualGraphs
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch_geometric.datasets import FakeDataset

from poptorch_geometric.batch_sampler import FixedBatchSampler, make_fixed_batch_generator
from poptorch_geometric.collate import CombinedBatchingCollater, FixedSizeCollater
from poptorch_geometric.dataloader import create_fixed_batch_dataloader


def test_fixed_batch_sampler_default_params():
    num_graphs = 10
    dataset = FakeDataset(num_graphs=num_graphs,
                          avg_num_nodes=30,
                          avg_degree=5)
    sampler = FixedBatchSampler(dataset, num_graphs=1)
    length = sum(1 for _ in itertools.chain(sampler))

    assert length == num_graphs


def test_fixed_batch_sampler_should_throw_exception():
    num_graphs = 3
    dataset = FakeDataset(num_graphs=num_graphs,
                          avg_num_nodes=30,
                          avg_degree=5)
    sampler = FixedBatchSampler(dataset,
                                num_graphs=2,
                                num_nodes=2,
                                allow_skip_data=False)
    with pytest.raises(RuntimeError):
        samples = []
        for sample in sampler:
            samples.append(sample)

    sampler = FixedBatchSampler(dataset,
                                num_graphs=2,
                                num_edges=2,
                                allow_skip_data=False)
    with pytest.raises(RuntimeError):
        samples = []
        for sample in sampler:
            samples.append(sample)


def test_fixed_batch_sampler_should_not_throw_exception():
    num_graphs = 4
    dataset = FakeDataset(num_graphs=num_graphs,
                          avg_num_nodes=30,
                          avg_degree=5)
    sampler = FixedBatchSampler(dataset,
                                num_graphs=2,
                                num_nodes=2,
                                allow_skip_data=True)
    length = sum(1 for _ in sampler)
    assert length == 0

    sampler = FixedBatchSampler(dataset,
                                num_graphs=2,
                                num_edges=2,
                                allow_skip_data=True)
    length = sum(1 for _ in sampler)
    assert length == 0


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("batch_num_graphs", [2, 10])
@pytest.mark.parametrize("allow_skip_data", [True, False])
def test_fixed_batch_should_return_valid_samples(shuffle, batch_num_graphs,
                                                 allow_skip_data):
    avg_num_nodes = 30
    dataset = FakeDataset(num_graphs=100,
                          avg_num_nodes=avg_num_nodes,
                          avg_degree=5,
                          num_channels=16,
                          edge_dim=8)
    avg_num_edges = int(math.ceil(mean((data.num_edges for data in dataset))))

    batch_num_nodes = avg_num_nodes * batch_num_graphs + batch_num_graphs
    if not allow_skip_data:
        max_num_nodes = max(data.num_nodes for data in dataset)
        batch_num_nodes = max(batch_num_nodes,
                              max_num_nodes + batch_num_graphs)

    batch_num_edges = avg_num_edges * batch_num_graphs + batch_num_graphs
    if not allow_skip_data:
        max_num_edges = max(data.num_edges for data in dataset)
        batch_num_edges = max(batch_num_edges,
                              max_num_edges + batch_num_graphs)

    base_sampler = RandomSampler(dataset) if shuffle else \
        SequentialSampler(dataset)

    # Leave space for padding.
    sampler = FixedBatchSampler(dataset,
                                num_graphs=batch_num_graphs - 1,
                                num_nodes=batch_num_nodes - 1,
                                num_edges=batch_num_edges - 1,
                                sampler=base_sampler,
                                allow_skip_data=allow_skip_data)
    length = sum(1 for _ in sampler)
    assert length > 0
    if not allow_skip_data:
        for sample in sampler:
            assert len(sample) <= batch_num_graphs

    batch_generator = make_fixed_batch_generator(sampler)

    for batch in batch_generator:
        assert batch.num_graphs == batch_num_graphs
        assert sum(batch[i].num_nodes
                   for i in range(batch.num_graphs)) == batch_num_nodes
        assert sum(batch[i].num_edges
                   for i in range(batch.num_graphs)) == batch_num_edges


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("allow_skip_data", [True, False])
@pytest.mark.parametrize("torch_data_loader", [True, False])
def test_fixed_batch_sampler_should_be_usable_with_torch_data_loader(
        shuffle, allow_skip_data, torch_data_loader):

    avg_num_nodes = 30
    batch_num_graphs = 10
    num_channels = 16
    edge_dim = 8
    num_graphs = 10
    dataset = FakeDataset(num_graphs=100,
                          avg_num_nodes=avg_num_nodes,
                          avg_degree=5,
                          num_channels=num_channels,
                          edge_dim=8)

    avg_num_edges = math.ceil(mean(data.num_edges for data in dataset))

    base_sampler = RandomSampler(dataset) if shuffle else \
        SequentialSampler(dataset)

    batch_num_nodes = avg_num_nodes * batch_num_graphs + batch_num_graphs
    if not allow_skip_data:
        max_num_nodes = max(data.num_nodes for data in dataset)
        batch_num_nodes = max(batch_num_nodes,
                              max_num_nodes + batch_num_graphs)

    batch_num_edges = avg_num_edges * batch_num_graphs + batch_num_graphs
    if not allow_skip_data:
        max_num_edges = max(data.num_edges for data in dataset)
        batch_num_edges = max(batch_num_edges,
                              max_num_edges + batch_num_graphs)

    # Leave space for padding.
    if torch_data_loader:
        batch_sampler = FixedBatchSampler(dataset,
                                          num_graphs=num_graphs - 1,
                                          num_nodes=batch_num_nodes - 1,
                                          num_edges=batch_num_edges - 1,
                                          sampler=base_sampler,
                                          allow_skip_data=allow_skip_data)

        collater = CombinedBatchingCollater(
            FixedSizeCollater(batch_num_nodes,
                              batch_num_edges,
                              batch_num_graphs,
                              node_pad_value=0.0,
                              edge_pad_value=0.0,
                              graph_pad_value=0.0,
                              add_masks_to_batch=True))

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_sampler=batch_sampler,
                                                 collate_fn=collater)
    else:
        collater_args = {
            "add_masks_to_batch": True,
            "num_edges": batch_num_edges,
            "num_graphs": num_graphs,
            "node_pad_value": 0.0,
            "edge_pad_value": 0.0,
            "graph_pad_value": 0.0
        }
        dataloader = create_fixed_batch_dataloader(
            dataset,
            num_nodes=batch_num_nodes,
            num_edges=batch_num_edges,
            batch_size=num_graphs,
            collater_args=collater_args,
            sampler=base_sampler,
            allow_skip_data=allow_skip_data)

    expected_x_shape = torch.Size([1, batch_num_nodes, num_channels])
    expected_batch_shape = torch.Size([1, batch_num_nodes])
    expected_edge_attr_shape = torch.Size([1, batch_num_edges, edge_dim])
    expected_mask_attr_shape = torch.Size([1, batch_num_graphs])
    expected_edge_index_attr_shape = torch.Size([1, 2, batch_num_edges])

    for data in dataloader:
        assert data.x.shape == expected_x_shape
        assert data.batch.shape == expected_batch_shape
        assert data.edge_attr.shape == expected_edge_attr_shape
        assert data.graphs_mask.shape == expected_mask_attr_shape
        assert data.edge_index.shape == expected_edge_index_attr_shape


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("allow_skip_data", [True, False])
def test_fixed_batch_sampler_padding_not_needed(shuffle, allow_skip_data):

    num_graphs_in_dataset = 100
    num_nodes = 30
    batch_num_graphs = 10
    num_channels = 16
    edge_dim = 8

    dataset = FakeDatasetEqualGraphs(num_graphs=num_graphs_in_dataset,
                                     num_nodes=num_nodes,
                                     num_channels=num_channels,
                                     edge_dim=edge_dim)

    avg_num_edges = math.ceil(mean(data.num_edges for data in dataset))

    base_sampler = RandomSampler(dataset) if shuffle else \
        SequentialSampler(dataset)

    batch_num_nodes = num_nodes * batch_num_graphs
    if not allow_skip_data:
        max_num_nodes = max(data.num_nodes for data in dataset)
        batch_num_nodes = max(batch_num_nodes,
                              max_num_nodes + batch_num_graphs)

    batch_num_edges = avg_num_edges * batch_num_graphs
    if not allow_skip_data:
        max_num_edges = max(data.num_edges for data in dataset)
        batch_num_edges = max(batch_num_edges,
                              max_num_edges + batch_num_graphs)

    batch_sampler = FixedBatchSampler(dataset,
                                      num_graphs=batch_num_graphs,
                                      num_nodes=batch_num_nodes,
                                      num_edges=batch_num_edges,
                                      sampler=base_sampler,
                                      allow_skip_data=allow_skip_data)

    collator = CombinedBatchingCollater(
        FixedSizeCollater(batch_num_nodes,
                          batch_num_edges,
                          batch_num_graphs,
                          node_pad_value=0.0,
                          edge_pad_value=0.0,
                          graph_pad_value=0.0,
                          add_masks_to_batch=True))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_sampler=batch_sampler,
                                             collate_fn=collator)

    expected_x_shape = torch.Size([1, batch_num_nodes, num_channels])
    expected_batch_shape = torch.Size([1, batch_num_nodes])
    expected_edge_attr_shape = torch.Size([1, batch_num_edges, edge_dim])
    expected_mask_attr_shape = torch.Size([1, batch_num_graphs])
    expected_edge_index_attr_shape = torch.Size([1, 2, batch_num_edges])

    total_graphs_from_dataloader = 0
    for data in dataloader:
        assert data.x.shape == expected_x_shape
        assert data.batch.shape == expected_batch_shape
        assert data.edge_attr.shape == expected_edge_attr_shape
        assert data.graphs_mask.shape == expected_mask_attr_shape
        assert data.edge_index.shape == expected_edge_index_attr_shape
        total_graphs_from_dataloader += data.num_graphs

    assert total_graphs_from_dataloader == num_graphs_in_dataset


def test_make_fixed_batch_generator_incorrect_values():
    dataset = FakeDataset(num_graphs=10, avg_num_nodes=10, avg_degree=3)

    sampler = FixedBatchSampler(dataset, num_graphs=3)
    batch_generator = make_fixed_batch_generator(sampler,
                                                 dataset,
                                                 num_graphs=3)
    with pytest.raises(ValueError, match=r'parameter `num_graphs` \(= 3\) ' \
                                         r'should be greater'):
        next(iter(batch_generator))

    sampler = FixedBatchSampler(dataset, num_graphs=2, num_nodes=20)
    batch_generator = make_fixed_batch_generator(sampler,
                                                 dataset,
                                                 num_graphs=3,
                                                 num_nodes=20)
    with pytest.raises(ValueError, match=r'parameter `num_nodes` \(= 20\) ' \
                                         r'should be greater'):
        next(iter(batch_generator))

    sampler = FixedBatchSampler(dataset,
                                num_graphs=2,
                                num_nodes=20,
                                num_edges=50)
    batch_generator = make_fixed_batch_generator(sampler,
                                                 dataset,
                                                 num_graphs=3,
                                                 num_nodes=25,
                                                 num_edges=50)
    with pytest.raises(ValueError, match=r'parameter `num_edges` \(= 50\) ' \
                                         r'should be greater'):
        next(iter(batch_generator))
