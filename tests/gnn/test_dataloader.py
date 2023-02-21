# Copyright (c) 2022-2023 Graphcore Ltd. All rights reserved.
import inspect
import pickle

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import FakeDataset
from torch_geometric.transforms import Pad

import utils
from poptorch_geometric.batch_sampler import FixedBatchSampler
from poptorch_geometric.collate import CombinedBatchingCollater, make_exclude_keys
from poptorch_geometric.dataloader import DataLoader as IPUDataLoader
from poptorch_geometric.dataloader import CustomFixedSizeDataLoader as IPUCustomFixedSizeDataLoader
from poptorch_geometric.dataloader import \
    FixedSizeDataLoader as IPUFixedSizeDataLoader
from poptorch_geometric.pyg_collate import Collater
from poptorch_geometric.pyg_dataloader import (CustomFixedSizeDataLoader,
                                               DataLoader, FixedSizeDataLoader)
from poptorch_geometric.types import PyGArgsParser
import poptorch


def _compare_batches(batch_actual: Batch, batch_expected: Batch):
    for key in batch_expected.keys:
        expected_value = batch_expected[key]
        actual_value = batch_actual[key]
        if isinstance(expected_value, torch.Tensor):
            assert torch.equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def test_batch_serialization():
    data = FakeDataset(num_graphs=1, avg_num_nodes=30, avg_degree=5)[0]
    batch = Batch.from_data_list([data])
    serialized_batch = pickle.dumps(batch)
    batch_unserialized = pickle.loads(serialized_batch)
    _compare_batches(batch_unserialized, batch)


def test_custom_batch_parser():
    data = FakeDataset(num_graphs=1, avg_num_nodes=30, avg_degree=5)[0]
    batch = Batch.from_data_list([data])
    parser = PyGArgsParser()
    generator = parser.yieldTensors(batch)
    batch_reconstructed = parser.reconstruct(batch, generator)
    _compare_batches(batch_reconstructed, batch)


def test_collater(molecule):
    include_keys = ('x', 'y', 'z')
    exclude_keys = make_exclude_keys(include_keys, molecule)
    collate_fn = Collater(exclude_keys=exclude_keys)
    batch = collate_fn([molecule])
    assert isinstance(batch, type(Batch(_base_cls=Data().__class__)))
    batch_keys = list(
        filter(lambda key: key not in ("ptr", "batch"), batch.keys))

    assert len(batch_keys) == len(include_keys)

    for key in include_keys:
        utils.assert_equal(actual=batch[key], expected=getattr(molecule, key))
        utils.assert_equal(actual=getattr(batch, key),
                           expected=getattr(molecule, key))


def test_multiple_collater(molecule):
    r"""Test that we can have two different collaters at the same time and
    that attribute access works as expected."""
    include_keys = ('x', )
    exclude_keys = make_exclude_keys(include_keys, molecule)
    indclude_keys_2 = ('z', )
    exclude_keys_2 = make_exclude_keys(indclude_keys_2, molecule)
    batch = Collater(exclude_keys=exclude_keys)([molecule])
    batch_2 = Collater(exclude_keys=exclude_keys_2)([molecule])

    for k1, k2 in zip(include_keys, indclude_keys_2):
        assert k1 in batch.keys
        assert k2 in batch_2.keys
        assert k2 not in batch.keys
        assert k1 not in batch_2.keys


def test_collater_invalid_keys(molecule):
    exclude_keys = ('v', 'name')
    collate_fn = Collater(exclude_keys=exclude_keys)

    batch = collate_fn([molecule])
    assert isinstance(batch, type(Batch(_base_cls=Data().__class__)))
    batch_keys = list(
        filter(lambda key: key not in ("ptr", "batch"), batch.keys))

    expected_keys = ['edge_index', 'pos', 'y', 'idx', 'z', 'edge_attr', 'x']

    assert len(expected_keys) == len(batch_keys)

    for key in expected_keys:
        utils.assert_equal(actual=batch[key], expected=getattr(molecule, key))
        utils.assert_equal(actual=getattr(batch, key),
                           expected=getattr(molecule, key))


@pytest.mark.parametrize('mini_batch_size', [1, 16])
def test_combined_batching_collater(molecule, mini_batch_size):
    # Simulates 4 replicas.
    num_replicas = 4
    combined_batch_size = num_replicas * mini_batch_size
    data_list = [molecule] * combined_batch_size
    collate_fn = CombinedBatchingCollater(mini_batch_size=mini_batch_size,
                                          collater=Collater())
    batch = collate_fn(data_list)
    for _, v in batch.items():
        if isinstance(v, torch.Tensor):
            assert v.shape[0] == num_replicas


def test_combined_batching_collater_invalid(molecule):
    collate_fn = CombinedBatchingCollater(mini_batch_size=8,
                                          collater=Collater())

    with pytest.raises(AssertionError, match='Invalid batch size'):
        collate_fn([molecule] * 9)


def test_simple_fixed_size_data_loader_mro(num_graphs=2, num_nodes=30):
    # Check that MROs of the dataloader classes are correct. There are other
    # classes that inherit from `FixedSizeDataLoader` and would be
    # affected if the MRO changes here.
    dataset = FakeDataset(num_graphs=num_graphs, avg_num_nodes=30)

    pyg_dataloader = FixedSizeDataLoader(dataset,
                                         num_nodes=num_nodes,
                                         batch_size=num_graphs)

    mro = inspect.getmro(type(pyg_dataloader))
    # MRO is longer but it's enough to check these classes.
    expected_mro = (FixedSizeDataLoader, CustomFixedSizeDataLoader,
                    torch.utils.data.DataLoader)
    num_classes = len(expected_mro)
    assert mro[:num_classes] == expected_mro

    ipu_dataloader = IPUFixedSizeDataLoader(dataset=dataset,
                                            num_nodes=num_nodes,
                                            batch_size=num_graphs)
    mro = inspect.getmro(type(ipu_dataloader))
    # MRO is longer but it's enough to check these classes.
    expected_mro = (IPUFixedSizeDataLoader, FixedSizeDataLoader,
                    IPUCustomFixedSizeDataLoader, CustomFixedSizeDataLoader,
                    poptorch.DataLoader, torch.utils.data.DataLoader)
    num_classes = len(expected_mro)
    assert mro[:num_classes] == expected_mro


@pytest.mark.parametrize('loader', [
    CustomFixedSizeDataLoader,
    dict(loader_cls=IPUCustomFixedSizeDataLoader, device_iterations=3),
    dict(loader_cls=IPUCustomFixedSizeDataLoader)
])
@pytest.mark.parametrize('use_batch_sampler', [True, False])
def test_fixed_size_dataloader(
        loader,
        use_batch_sampler,
        benchmark,
        fake_molecular_dataset,
        batch_size=10,
):
    ipu_dataloader = loader is not CustomFixedSizeDataLoader
    # CombinedBatchingCollater adds an additional 0-th dimension.
    dim_offset = 1 if ipu_dataloader else 0

    # Get a sensible value for the the maximum number of nodes.
    padded_num_nodes = fake_molecular_dataset.avg_num_nodes * (batch_size + 10)
    padded_num_edges = fake_molecular_dataset.avg_degree * padded_num_nodes

    # Define the expected tensor sizes in the output.
    data = fake_molecular_dataset[0]
    data_attributes = (k for k, _ in data()
                       if data.is_node_attr(k) or data.is_edge_attr(k))
    expected_sizes = {
        k: ((padded_num_nodes if data.is_node_attr(k) else padded_num_edges),
            dim_offset)
        for k in data_attributes
    }
    # Special case for edge_index which is of shape [2, num_edges].
    expected_sizes['edge_index'] = (padded_num_edges, 1 + dim_offset)

    # Create a fixed size dataloader.
    kwargs = {
        'dataset': fake_molecular_dataset,
        'num_nodes': padded_num_nodes,
        'collater_args': {
            'num_edges': padded_num_edges,
        }
    }
    if use_batch_sampler:
        batch_sampler = FixedBatchSampler(fake_molecular_dataset,
                                          batch_size - 1,
                                          num_nodes=padded_num_nodes - 1,
                                          num_edges=padded_num_edges - 1)
        kwargs['batch_sampler'] = batch_sampler
    else:
        kwargs['batch_size'] = batch_size

    if ipu_dataloader:
        options = poptorch.Options()
        default_options_device_iterations = options.device_iterations
        device_iterations = loader.get('device_iterations',
                                       default_options_device_iterations)
        options.deviceIterations(device_iterations=device_iterations)
        kwargs['options'] = options
        loader = loader['loader_cls']

    loader = loader(**kwargs)

    # Check that each batch matches the expected size.
    for batch in loader:
        assert hasattr(batch, 'batch')
        assert hasattr(batch, 'ptr')

        if ipu_dataloader:
            assert list(
                batch.batch.size()) == [device_iterations, padded_num_nodes]
            if not use_batch_sampler:
                assert list(
                    batch.ptr.size()) == [device_iterations, batch_size + 1]
        else:
            assert list(batch.batch.size()) == [padded_num_nodes]
            if not use_batch_sampler:
                assert list(batch.ptr.size()) == [batch_size + 1]

        sizes_match = all(
            getattr(batch, k).shape[dim] == size
            for k, (size, dim) in expected_sizes.items())
        assert sizes_match

    def loop():
        for _ in loader:
            pass

    benchmark(loop)


@pytest.mark.parametrize('custom_loader', [True, False])
@pytest.mark.parametrize('num_edges', [None, 500])
@pytest.mark.parametrize('num_graphs', [2, 10])
def test_dataloader_produces_fixed_sizes(custom_loader, num_edges, num_graphs,
                                         fake_molecular_dataset):
    num_nodes = num_graphs * 30
    dataset_size = 123
    dataset = fake_molecular_dataset[:dataset_size]

    if custom_loader:
        train_dataloader = CustomFixedSizeDataLoader(dataset,
                                                     num_nodes=num_nodes,
                                                     batch_size=num_graphs,
                                                     collater_args={
                                                         'add_masks_to_batch':
                                                         True,
                                                         'num_edges':
                                                         num_edges,
                                                         'trim_nodes': True,
                                                         'trim_edges': True,
                                                     })
    else:
        train_dataloader = FixedSizeDataLoader(
            dataset,
            num_nodes=num_nodes,
            batch_size=num_graphs,
            num_edges=num_edges,
            collater_args={'add_masks_to_batch': True})

    batch = next(iter(train_dataloader))
    attrs = [
        attr for attr in batch.keys if isinstance(batch[attr], torch.Tensor)
    ]
    for data in train_dataloader:
        for attr in attrs:
            assert batch[attr].shape == data[attr].shape


def test_dataloader(
        fake_molecular_dataset,
        batch_size=10,
):
    dataset = fake_molecular_dataset
    loader = DataLoader(dataset=dataset, batch_size=batch_size)

    for idx, batch in enumerate(loader):
        assert hasattr(batch, 'batch')
        assert hasattr(batch, 'ptr')

        # Check that each batch matches the expected size.
        idx_range = slice(idx * batch_size, (idx + 1) * batch_size)
        assert batch.num_graphs == batch_size
        assert batch.num_nodes == sum(d.num_nodes for d in dataset[idx_range])
        assert batch.num_edges == sum(d.num_edges for d in dataset[idx_range])

        # Split batch to the list of data and compare with the data from the
        # dataset.
        data_list = batch.to_data_list()
        for original, new in zip(dataset[idx_range], data_list):

            assert set(new.keys) == set(original.keys)

            for key in original.keys:
                if not isinstance(original[key], torch.Tensor):
                    assert new[key] == original[key]
                else:
                    assert torch.all(torch.eq(new[key], original[key]))


@pytest.mark.parametrize('device_iterations', [None, 3])
def test_padded_dataloader(
        device_iterations,
        fake_molecular_dataset,
        batch_size=10,
):
    dataset = fake_molecular_dataset[:123]
    dataset.transform = Pad(max_num_nodes=30, max_num_edges=150)

    options = poptorch.Options()
    if device_iterations is not None:
        options.deviceIterations(device_iterations=device_iterations)

    loader = IPUDataLoader(dataset=dataset,
                           batch_size=batch_size,
                           options=options)

    # Create PyG's dataloader to compare the created batches.
    pyg_loader = DataLoader(dataset=dataset, batch_size=batch_size)
    torch_loader_iter = iter(pyg_loader)

    for idx, batch in enumerate(loader):
        assert hasattr(batch, 'batch')
        assert hasattr(batch, 'ptr')

        # Check that each batch matches the expected size.
        idx_range = slice(idx * batch_size, (idx + 1) * batch_size)
        assert batch.num_graphs == batch_size
        assert batch.num_nodes == sum(d.num_nodes for d in dataset[idx_range])
        assert batch.num_edges == sum(d.num_edges for d in dataset[idx_range])

        num_batches = device_iterations or 1

        for key in batch.keys:
            if isinstance(batch[key], torch.Tensor):
                assert batch[key].shape[0] == num_batches

        # Compare batches from PyG's and PopPyG's dataloaders.
        torch_batches = [next(torch_loader_iter) for _ in range(num_batches)]
        for b_idx, torch_batch in enumerate(torch_batches):
            assert set(batch.keys) == set(torch_batch.keys)

            for key in torch_batch.keys:
                if not isinstance(torch_batch[key], torch.Tensor):
                    assert batch[key] == torch_batch[key]
                else:
                    assert torch.all(
                        torch.eq(batch[key][b_idx, ], torch_batch[key]))


@pytest.mark.parametrize('allow_skip_data', [True, False])
def test_dataloader_with_sampler_num_nodes(allow_skip_data,
                                           fake_molecular_dataset):
    dataset = fake_molecular_dataset[:10]

    sampler = FixedBatchSampler(dataset,
                                num_graphs=1,
                                num_nodes=100,
                                allow_skip_data=allow_skip_data)

    with pytest.raises(AssertionError,
                       match=r'Argument `num_nodes` \(= 100\) should ' \
                             r'be greater'):
        dataloader = CustomFixedSizeDataLoader(dataset,
                                               batch_sampler=sampler,
                                               num_nodes=100)

    num_nodes = 101
    dataloader = CustomFixedSizeDataLoader(dataset,
                                           batch_sampler=sampler,
                                           num_nodes=num_nodes)

    for batch in dataloader:
        assert batch.num_nodes == num_nodes


@pytest.mark.parametrize('create_loader',
                         [FixedSizeDataLoader, IPUFixedSizeDataLoader])
def test_fixed_size_dataloader_num_created_batches(create_loader):
    total_num_graphs = 100
    ds = FakeDataset(num_graphs=total_num_graphs, avg_num_nodes=10)
    total_num_nodes = sum(d.num_nodes for d in ds)
    total_num_edges = sum(d.num_edges for d in ds)

    # Loader should create 10 batches of 11 graphs each (10 real + 1 padding
    # graph).
    expected_num_batches = 10
    padded_batch_size = 11
    loader = create_loader(ds,
                           batch_size=padded_batch_size,
                           num_nodes=total_num_nodes)
    batches_created = sum(1 for _ in loader)

    assert batches_created == expected_num_batches

    # Loader should create only 1 batch since there is space for all graphs
    # and one padding graph.
    expected_num_batches = 1
    loader = create_loader(ds,
                           batch_size=101,
                           num_nodes=total_num_nodes + 1,
                           num_edges=total_num_edges + 1)
    batches_created = sum(1 for _ in loader)

    assert batches_created == expected_num_batches

    # There is no space for padding graph in the first batch (not enough
    # graphs) so loader should create two batches.
    expected_num_batches = 2
    loader = create_loader(ds,
                           batch_size=100,
                           num_nodes=total_num_nodes + 1,
                           num_edges=total_num_edges + 1)
    batches_created = sum(1 for _ in loader)

    assert batches_created == expected_num_batches

    # There is no space for padding graph in the first batch (not enough
    # nodes) so loader should create two batches.
    expected_num_batches = 2
    loader = create_loader(ds,
                           batch_size=101,
                           num_nodes=total_num_nodes,
                           num_edges=total_num_edges + 1)
    batches_created = sum(1 for _ in loader)

    assert batches_created == expected_num_batches

    # There is no space for padding graph in the first batch (not enough
    # edges) so loader should create two batches.
    expected_num_batches = 2
    loader = create_loader(ds,
                           batch_size=101,
                           num_nodes=total_num_nodes + 1,
                           num_edges=total_num_edges)
    batches_created = sum(1 for _ in loader)

    assert batches_created == expected_num_batches


def test_num_nodes_default_value():
    total_num_graphs = 100
    ds = FakeDataset(num_graphs=total_num_graphs, avg_num_nodes=10)

    batch_size = 10
    padded_batch_size = batch_size + 1
    # The default value of `num_nodes` should be large enough so it's possible
    # to always pick 10 graphs and create additional padding graph.
    loader = CustomFixedSizeDataLoader(ds, batch_size=padded_batch_size)
    expected_batches = 10

    num_batches = sum(1 for _ in loader)
    assert expected_batches == num_batches

    # The same when using FixedSizeDataLoader.
    loader = FixedSizeDataLoader(ds, batch_size=padded_batch_size)

    num_batches = sum(1 for _ in loader)
    assert expected_batches == num_batches

    # DataLoader should correctly capture the number of nodes from sampler.
    sampler = FixedBatchSampler(ds, num_graphs=batch_size)
    loader = CustomFixedSizeDataLoader(ds, batch_sampler=sampler)

    num_batches = 0
    for batch in loader:
        assert batch.num_nodes == sampler.num_nodes + 1
        num_batches += 1
    assert expected_batches == num_batches
