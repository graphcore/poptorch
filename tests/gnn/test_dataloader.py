# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import inspect
import pickle

from functools import singledispatch

import pytest
import torch
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.datasets import FakeDataset
from torch_geometric.transforms import Pad

import utils
from utils import is_data
from poptorch_geometric.stream_packing_sampler import StreamPackingSampler
from poptorch_geometric.collate import CombinedBatchingCollater, make_exclude_keys
from poptorch_geometric.dataloader import DataLoader as IPUDataLoader
from poptorch_geometric.dataloader import \
    FixedSizeDataLoader as IPUFixedSizeDataLoader
from poptorch_geometric.dataloader import FixedSizeStrategy, OverSizeStrategy
from poptorch_geometric.fixed_size_options import FixedSizeOptions
from poptorch_geometric.pyg_collate import Collater
from poptorch_geometric.pyg_dataloader import (DataLoader, FixedSizeDataLoader)
from poptorch_geometric.types import PyGArgsParser
from poptorch_geometric.common import DataBatch, HeteroDataBatch

import poptorch

# pylint: disable=protected-access


@singledispatch
def _compare_batches(batch_actual, batch_expected):
    raise ValueError(f'Unsupported data type: {type(batch_actual)}')


@_compare_batches.register
def _(batch_actual: DataBatch, batch_expected: DataBatch):
    for key in batch_expected.keys:
        expected_value = batch_expected[key]
        actual_value = batch_actual[key]
        if isinstance(expected_value, torch.Tensor):
            assert torch.equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


@_compare_batches.register
def _(batch_actual: HeteroDataBatch, batch_expected: HeteroDataBatch):
    for actual, expected in zip(batch_actual._global_store.values(),
                                batch_expected._global_store.values()):
        assert actual == expected

    def compare_stores(actual, expected):
        for a, e in zip(actual, expected):
            for act, exp in zip(a.values(), e.values()):
                assert act.tolist() == exp.tolist()

    compare_stores(batch_actual.node_stores, batch_expected.node_stores)
    compare_stores(batch_actual.edge_stores, batch_expected.edge_stores)


@pytest.mark.parametrize('dataset',
                         ['fake_small_dataset', 'fake_hetero_dataset'])
def test_batch_serialization(dataset, request):
    dataset = request.getfixturevalue(dataset)
    data = dataset[0]
    batch = Batch.from_data_list([data])
    serialized_batch = pickle.dumps(batch)
    batch_unserialized = pickle.loads(serialized_batch)
    _compare_batches(batch_unserialized, batch)


@pytest.mark.parametrize('dataset',
                         ['fake_small_dataset', 'fake_hetero_dataset'])
def test_custom_batch_parser(dataset, request):
    dataset = request.getfixturevalue(dataset)
    data = dataset[0]
    batch = Batch.from_data_list([data])
    parser = PyGArgsParser()
    generator = parser.yieldTensors(batch)
    batch_reconstructed = parser.reconstruct(batch, generator)
    _compare_batches(batch_reconstructed, batch)


@pytest.mark.parametrize('data', ['molecule', 'fake_hetero_data'])
def test_collater(data, request):
    data = request.getfixturevalue(data)
    if isinstance(data, Data):
        include_keys = ('x', 'y', 'z')
    else:
        include_keys = ('x')

    exclude_keys = make_exclude_keys(include_keys, data)
    collate_fn = Collater(exclude_keys=exclude_keys)
    batch = collate_fn([data])
    data_type = type(data)
    assert isinstance(batch, type(Batch(_base_cls=data_type)))
    batch_keys = list(
        filter(lambda key: key not in ('ptr', 'batch', 'edge_index'),
               batch.keys))

    assert len(batch_keys) == len(include_keys)

    for key in include_keys:
        if is_data(data_type):
            utils.assert_equal(actual=batch[key], expected=getattr(data, key))
            utils.assert_equal(actual=getattr(batch, key),
                               expected=getattr(data, key))
        else:
            for b_store, d_store in zip(batch.node_stores, data.node_stores):
                utils.assert_equal(actual=b_store[key],
                                   expected=getattr(d_store, key))
                utils.assert_equal(actual=getattr(b_store, key),
                                   expected=getattr(d_store, key))


@pytest.mark.parametrize('data', ['molecule', 'fake_hetero_data'])
def test_multiple_collater(data, request):
    r"""Test that we can have two different collaters at the same time and
    that attribute access works as expected."""
    data = request.getfixturevalue(data)

    include_keys = ('x', )
    exclude_keys = make_exclude_keys(include_keys, data)
    indclude_keys_2 = ('z', )
    exclude_keys_2 = make_exclude_keys(indclude_keys_2, data)
    batch = Collater(exclude_keys=exclude_keys)([data])
    batch_2 = Collater(exclude_keys=exclude_keys_2)([data])

    for k1, k2 in zip(include_keys, indclude_keys_2):
        assert k1 in batch.keys
        assert k2 not in batch.keys
        assert k1 not in batch_2.keys
        if is_data(type(data)):
            assert k2 in batch_2.keys


@pytest.mark.parametrize('data', ['molecule', 'fake_hetero_data'])
def test_collater_invalid_keys(data, request):
    data = request.getfixturevalue(data)
    if not isinstance(data, Data):
        data['y'] = torch.zeros(1)
        expected_keys = ['edge_index', 'x', 'y']
    else:
        expected_keys = [
            'edge_index', 'pos', 'y', 'idx', 'z', 'edge_attr', 'x'
        ]

    data_type = type(data)

    exclude_keys = ('v', 'name')
    collate_fn = Collater(exclude_keys=exclude_keys)

    batch = collate_fn([data])
    assert isinstance(batch, type(Batch(_base_cls=data_type)))
    batch_keys = list(
        filter(lambda key: key not in ('ptr', 'batch'), batch.keys))

    assert len(expected_keys) == len(batch_keys)
    if is_data(data_type):
        for key in expected_keys:
            utils.assert_equal(actual=batch[key], expected=getattr(data, key))
            utils.assert_equal(actual=getattr(batch, key),
                               expected=getattr(data, key))
    else:

        def check(batch_stores, data_stores, key):
            for b_store, d_store in zip(batch_stores, data_stores):
                utils.assert_equal(actual=b_store[key],
                                   expected=getattr(d_store, key))
                utils.assert_equal(actual=getattr(b_store, key),
                                   expected=getattr(d_store, key))

        key = 'edge_index'
        check(batch.edge_stores, data.edge_stores, key)
        key = 'x'
        check(batch.node_stores, data.node_stores, key)
        key = 'y'
        check((batch._global_store, ), (data._global_store, ), key)


@pytest.mark.parametrize('data', ['molecule', 'fake_hetero_data'])
@pytest.mark.parametrize('mini_batch_size', [1, 16])
def test_combined_batching_collater(mini_batch_size, data, request):
    data = request.getfixturevalue(data)

    # Simulates 4 replicas.
    num_replicas = 4
    combined_batch_size = num_replicas * mini_batch_size
    data_list = [data] * combined_batch_size
    collate_fn = CombinedBatchingCollater(mini_batch_size=mini_batch_size,
                                          collater=Collater())
    batch = collate_fn(data_list)
    for key, v in batch.items():
        if isinstance(v, torch.Tensor):
            if key == 'batch':
                size = sum(d.num_nodes for d in data_list)
                assert v.shape[0] == size
            elif key == 'ptr':
                assert v.shape[0] == (mini_batch_size + 1) * num_replicas
            else:
                if key == 'edge_index':
                    assert v.shape[0] == num_replicas * 2
                    assert v.shape[
                        1] == data.edge_index.shape[1] * mini_batch_size
                else:
                    size = sum(d[key].shape[0] for d in data_list)
                    assert v.shape[0] == size


def test_combined_batching_collater_invalid(molecule):
    collate_fn = CombinedBatchingCollater(mini_batch_size=8,
                                          collater=Collater())

    with pytest.raises(AssertionError, match='Invalid batch size'):
        collate_fn([molecule] * 9)


def test_simple_fixed_size_data_loader_mro(num_graphs=2, num_nodes=40):
    # Check that MROs of the dataloader classes are correct. There are other
    # classes that inherit from `FixedSizeDataLoader` and would be
    # affected if the MRO changes here.
    dataset = FakeDataset(num_graphs=num_graphs, avg_num_nodes=30)

    fixed_size_options = FixedSizeOptions(num_nodes=num_nodes,
                                          num_graphs=num_graphs)

    pyg_dataloader = FixedSizeDataLoader(dataset,
                                         fixed_size_options=fixed_size_options,
                                         batch_size=num_graphs)

    mro = inspect.getmro(type(pyg_dataloader))
    # MRO is longer but it's enough to check these classes.
    expected_mro = (FixedSizeDataLoader, torch.utils.data.DataLoader)
    num_classes = len(expected_mro)
    assert mro[:num_classes] == expected_mro

    ipu_dataloader = IPUFixedSizeDataLoader(
        dataset=dataset,
        fixed_size_options=fixed_size_options,
        batch_size=num_graphs)
    mro = inspect.getmro(type(ipu_dataloader))
    # MRO is longer but it's enough to check these classes.
    expected_mro = (IPUFixedSizeDataLoader, FixedSizeDataLoader,
                    poptorch.DataLoader, torch.utils.data.DataLoader)
    num_classes = len(expected_mro)
    assert mro[:num_classes] == expected_mro


@pytest.mark.parametrize('loader', [
    FixedSizeDataLoader,
    dict(loader_cls=IPUFixedSizeDataLoader, device_iterations=3),
    dict(loader_cls=IPUFixedSizeDataLoader)
])
@pytest.mark.parametrize(
    'fixed_size_strategy',
    [FixedSizeStrategy.PadToMax, FixedSizeStrategy.StreamPack])
@pytest.mark.parametrize('dataset', ['pyg_qm9', 'fake_node_task_dataset'])
def test_fixed_size_dataloader(loader,
                               fixed_size_strategy,
                               benchmark,
                               dataset,
                               request,
                               batch_size=10):
    dataset = request.getfixturevalue(dataset)

    ipu_dataloader = loader is not FixedSizeDataLoader
    # CombinedBatchingCollater adds an additional 0-th dimension.
    dim_offset = 0

    device_iterations = loader.get(
        'device_iterations',
        poptorch.Options().device_iterations) if ipu_dataloader else 1

    # Get a sensible value for the the maximum number of nodes.
    padded_num_nodes = dataset[0].num_nodes * (batch_size + 20)
    padded_num_edges = dataset[0].num_edges * padded_num_nodes

    # Define the expected tensor sizes in the output.
    data = dataset[0]
    data_attributes = (k for k, _ in data()
                       if data.is_node_attr(k) or data.is_edge_attr(k))
    expected_sizes = {
        k: ((padded_num_nodes if data.is_node_attr(k) else padded_num_edges) *
            device_iterations, dim_offset)
        for k in data_attributes
    }
    # Special case for edge_index which is of shape [2, num_edges].
    expected_sizes['edge_index'] = (device_iterations * 2, dim_offset)

    # Special case for `y` being graph-lvl label
    if not data.is_node_attr('y'):
        expected_sizes['y'] = (batch_size * device_iterations, dim_offset)

    # Create a fixed size dataloader.
    kwargs = {
        'dataset':
        dataset,
        'batch_size':
        batch_size,
        'fixed_size_options':
        FixedSizeOptions(num_nodes=padded_num_nodes,
                         num_edges=padded_num_edges,
                         num_graphs=batch_size),
        'fixed_size_strategy':
        fixed_size_strategy
    }

    if ipu_dataloader:
        options = poptorch.Options()
        options.deviceIterations(device_iterations=device_iterations)
        kwargs['options'] = options
        loader = loader['loader_cls']

    loader = loader(**kwargs)

    # Check that each batch matches the expected size.
    loader_iter = iter(loader)
    repeats = 10
    for _ in range(repeats):
        batch = next(loader_iter)
        assert hasattr(batch, 'batch')
        assert hasattr(batch, 'ptr')

        if ipu_dataloader:
            assert list(batch.batch.size()) == [
                device_iterations * padded_num_nodes,
            ]
            if not fixed_size_strategy == FixedSizeStrategy.StreamPack:
                assert list(batch.ptr.size()) == [
                    device_iterations * (batch_size + 1),
                ]
        else:
            assert list(batch.batch.size()) == [padded_num_nodes]
            if not fixed_size_strategy == FixedSizeStrategy.StreamPack:
                assert list(batch.ptr.size()) == [batch_size + 1]

        sizes_match = all(
            getattr(batch, k).shape[dim] == size
            for k, (size, dim) in expected_sizes.items())
        assert sizes_match

    def loop():
        loader_iter = iter(loader)
        for _ in range(repeats):
            next(loader_iter)

    benchmark(loop)


@pytest.mark.parametrize('loader', [
    FixedSizeDataLoader,
    dict(loader_cls=IPUFixedSizeDataLoader, device_iterations=3),
    dict(loader_cls=IPUFixedSizeDataLoader)
])
@pytest.mark.parametrize(
    'fixed_size_strategy',
    [FixedSizeStrategy.PadToMax, FixedSizeStrategy.StreamPack])
@pytest.mark.parametrize(
    'dataset', ['fake_hetero_dataset', 'fake_node_task_hetero_dataset'])
@pytest.mark.parametrize('fixed_size_options,requires_trimming',
                         [(FixedSizeOptions(
                             num_nodes={
                                 "v0": 500,
                                 "v1": 1000,
                             },
                             num_edges={
                                 ("v0", "e0", "v1"): 5000,
                                 ("v0", "e0", "v0"): 6000,
                                 ("v1", "e0", "v0"): 7000,
                                 ("v0", "e1", "v1"): 8000,
                                 ("v1", "e0", "v1"): 9000,
                             },
                             num_graphs=10,
                         ), False),
                          (FixedSizeOptions(
                              num_nodes=1000,
                              num_edges={
                                  ("v0", "e0", "v1"): 5000,
                                  ("v0", "e0", "v0"): 6000,
                                  ("v1", "e0", "v0"): 7000,
                                  ("v0", "e1", "v1"): 8000,
                                  ("v1", "e0", "v1"): 9000,
                              },
                              num_graphs=10,
                          ), False),
                          (FixedSizeOptions(
                              num_nodes={
                                  "v0": 500,
                                  "v1": 1000,
                              },
                              num_edges=8000,
                              num_graphs=10,
                          ), False),
                          (FixedSizeOptions(
                              num_nodes={
                                  "v0": 100,
                                  "v1": 200,
                              },
                              num_edges={
                                  ("v0", "e0", "v1"): 2000,
                                  ("v0", "e0", "v0"): 300,
                                  ("v1", "e0", "v0"): 1000,
                                  ("v0", "e1", "v1"): 100,
                                  ("v1", "e0", "v1"): 3000,
                              },
                              num_graphs=10,
                          ), True)])
def test_fixed_size_heterodataloader(
        loader,
        fixed_size_strategy,
        benchmark,
        dataset,
        fixed_size_options,
        requires_trimming,
        request,
):
    dataset = request.getfixturevalue(dataset)
    ipu_dataloader = loader is not FixedSizeDataLoader

    batch_size = fixed_size_options.num_graphs

    device_iterations = loader.get(
        'device_iterations',
        poptorch.Options().device_iterations) if ipu_dataloader else 1

    # Create a fixed size dataloader.
    kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'fixed_size_options': fixed_size_options,
        'fixed_size_strategy': fixed_size_strategy,
    }

    if ipu_dataloader:
        options = poptorch.Options()
        options.deviceIterations(device_iterations=device_iterations)
        kwargs['options'] = options
        loader = loader['loader_cls']

    fixed_size_loader = loader(**kwargs)

    if requires_trimming:
        with pytest.raises(RuntimeError):
            next(iter(fixed_size_loader))
        fixed_size_loader = loader(
            over_size_strategy=OverSizeStrategy.TrimNodesAndEdges, **kwargs)

    for batch in fixed_size_loader:
        for node_attr in filter(is_iterable, batch.node_stores):
            check_batch_and_ptr(node_attr)

        assert batch.num_nodes == fixed_size_options.total_num_nodes
        assert batch.num_edges == fixed_size_options.total_num_edges
        assert 'num_nodes' not in batch.node_types
        assert 'num_edges' not in batch.edge_types

        if 'y' in batch._node_store_dict.keys():
            assert batch.y.shape[0] == batch_size * device_iterations
        assert batch.graphs_mask.shape[0] == batch_size * device_iterations

        assert sum(node_attr.batch.shape[0]
                   for node_attr in filter(is_iterable, batch.node_stores)
                   ) == fixed_size_options.total_num_nodes * device_iterations
        if not fixed_size_strategy == FixedSizeStrategy.StreamPack:
            assert {
                node_attr.ptr.shape[0]
                for node_attr in filter(is_iterable, batch.node_stores)
            } == {device_iterations * (batch_size + 1)}

        # Check sizes for some of the items in the batch
        for node_type in fixed_size_options.num_nodes:
            assert batch[node_type].x.shape[0] == fixed_size_options.num_nodes[
                node_type] * device_iterations
            assert batch[node_type].batch.shape[
                0] == fixed_size_options.num_nodes[
                    node_type] * device_iterations
            assert batch[node_type].nodes_mask.shape[
                0] == fixed_size_options.num_nodes[
                    node_type] * device_iterations
        for edge_type in fixed_size_options.num_edges:
            # Checking num of edges with second dimension so it is not a multiple
            # of device iterations.
            assert batch[edge_type].edge_index.shape[
                1] == fixed_size_options.num_edges[edge_type]
            assert batch[edge_type].edges_mask.shape[
                0] == fixed_size_options.num_edges[
                    edge_type] * device_iterations

    def loop():
        for _ in fixed_size_loader:
            pass

    benchmark(loop)


@pytest.mark.parametrize('num_edges', [None, 500])
@pytest.mark.parametrize('num_graphs', [2, 10])
@pytest.mark.parametrize(
    'fixed_size_strategy',
    [FixedSizeStrategy.PadToMax, FixedSizeStrategy.StreamPack])
def test_dataloader_trims_to_fixed_sizes(num_edges, num_graphs,
                                         fixed_size_strategy,
                                         fake_molecular_dataset):
    num_nodes = num_graphs * 30
    dataset_size = 123
    dataset = fake_molecular_dataset[:dataset_size]

    fixed_size_options = FixedSizeOptions(num_nodes=num_nodes,
                                          num_edges=num_edges,
                                          num_graphs=num_graphs)

    train_dataloader = FixedSizeDataLoader(
        dataset,
        fixed_size_options=fixed_size_options,
        batch_size=num_graphs,
        fixed_size_strategy=fixed_size_strategy,
        over_size_strategy=OverSizeStrategy.TrimNodesAndEdges)

    batch = next(iter(train_dataloader))
    attrs = [
        attr for attr in batch.keys if isinstance(batch[attr], torch.Tensor)
    ]
    for data in train_dataloader:
        for attr in attrs:
            assert batch[attr].shape == data[attr].shape


def is_iterable(src):
    return hasattr(src, '__iter__')


def check_batch_and_ptr(src):
    assert 'batch' in src
    assert 'ptr' in src


@pytest.mark.parametrize('dataset',
                         ['fake_molecular_dataset', 'fake_hetero_dataset'])
def test_dataloader(dataset, request, batch_size=10):
    dataset = request.getfixturevalue(dataset)
    loader = DataLoader(dataset=dataset, batch_size=batch_size)

    for idx, batch in enumerate(loader):
        if isinstance(batch, HeteroDataBatch):
            for node_attr in filter(is_iterable, batch.node_stores):
                check_batch_and_ptr(node_attr)
        else:
            check_batch_and_ptr(batch)

        # Check that each batch matches the expected size.
        idx_range = slice(idx * batch_size, (idx + 1) * batch_size)
        assert batch.num_graphs == batch_size
        assert batch.num_nodes == sum(d.num_nodes for d in dataset[idx_range])
        assert batch.num_edges == sum(d.num_edges for d in dataset[idx_range])

        # Split batch to the list of data and compare with the data from the
        # dataset.
        data_list = batch.to_data_list()

        def check_data_types(original, new):
            if isinstance(original, torch.Tensor):
                assert original.dtype == new.dtype
            else:
                for o, n in zip(original.values(), new.values()):
                    check_data_types(o, n)

        for original, new in zip(dataset[idx_range], data_list):
            assert set(new.keys) == set(original.keys)

            for o, n in zip(original.to_dict().values(),
                            new.to_dict().values()):
                check_data_types(o, n)

            for key in original.keys:
                if not isinstance(original[key], torch.Tensor):
                    assert new[key] == original[key]
                else:
                    assert torch.all(torch.eq(new[key], original[key]))


@pytest.mark.parametrize('dataset',
                         ['fake_molecular_dataset', 'fake_hetero_dataset'])
@pytest.mark.parametrize('device_iterations', [None, 3])
def test_pad_transform_with_dataloader(
        device_iterations,
        dataset,
        request,
        batch_size=3,
):
    """Tests the pattern of using a Pad transform and a non-fixed-size
       data loader as an approach to achieve fixed size batches"""
    dataset = request.getfixturevalue(dataset)
    is_HeteroData = isinstance(dataset[0], HeteroData)
    if is_HeteroData:
        max_num_nodes = 300
        max_num_edges = 1500

        def check(b_idx, torch_batch, batch):
            for t, b in zip(torch_batch.node_stores, batch.node_stores):
                assert set(t.keys()) == set(b.keys())
                for key in t.keys():
                    if isinstance(t[key], torch.Tensor):
                        shape_dim = t[key].shape[0]
                        slc = slice(b_idx * shape_dim, (b_idx + 1) * shape_dim)
                        assert all((b[key][slc] == t[key]).tolist())
                    else:
                        assert b[key] == t[key]
    else:
        max_num_nodes = 30
        max_num_edges = 150
        dataset = dataset[:123]

        def check(b_idx, torch_batch, batch):
            assert set(torch_batch.keys).issubset(set(batch.keys))
            for key in torch_batch.keys:
                if isinstance(torch_batch[key], torch.Tensor):
                    shape_dim = torch_batch[key].shape[0]
                    slc = slice(b_idx * shape_dim, (b_idx + 1) * shape_dim)
                    if isinstance(batch[key], torch.Tensor):
                        assert all(
                            (batch[key][slc] == torch_batch[key]).tolist())
                    else:
                        assert sum(torch_batch[key].tolist()) == batch[key]
                else:
                    assert batch[key] == torch_batch[key]

    dataset.transform = Pad(max_num_nodes=max_num_nodes,
                            max_num_edges=max_num_edges)

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
        if is_HeteroData:
            for node_attr in filter(is_iterable, batch.node_stores):
                check_batch_and_ptr(node_attr)
        else:
            check_batch_and_ptr(batch)

        # Check that each batch matches the expected size.
        idx_range = slice(idx * batch_size, (idx + 1) * batch_size)
        assert batch.num_graphs == batch_size
        assert batch.num_nodes == sum(d.num_nodes for d in dataset[idx_range])
        assert batch.num_edges == sum(d.num_edges for d in dataset[idx_range])
        num_batches = device_iterations or 1

        # Compare batches from PyG's and PopPyG's dataloaders.
        torch_batches = [next(torch_loader_iter) for _ in range(num_batches)]

        for b_idx, torch_batch in enumerate(torch_batches):
            check(b_idx, torch_batch, batch)


@pytest.mark.parametrize('dataset',
                         ['fake_molecular_dataset', 'fake_hetero_dataset'])
@pytest.mark.parametrize('allow_skip_data', [True, False])
def test_dataloader_with_sampler_num_nodes(allow_skip_data, dataset, request):
    num_node_types = 2 if dataset == 'fake_hetero_dataset' else 1
    dataset = request.getfixturevalue(dataset)
    num_nodes = 1000
    if isinstance(dataset[0], Data):
        dataset = dataset[:10]
        num_nodes = 100

    sampler = StreamPackingSampler(dataset,
                                   max_num_graphs=1,
                                   max_num_nodes=num_nodes,
                                   allow_skip_data=allow_skip_data)

    num_nodes = num_nodes + 1

    fixed_size_options = FixedSizeOptions(num_nodes=num_nodes)

    dataloader = FixedSizeDataLoader(dataset,
                                     fixed_size_options=fixed_size_options,
                                     batch_sampler=sampler)

    for batch in dataloader:
        assert batch.num_nodes == num_nodes * num_node_types


@pytest.mark.parametrize('create_loader',
                         [FixedSizeDataLoader, IPUFixedSizeDataLoader])
def test_fixed_size_dataloader_num_created_batches_stream_packing(
        create_loader):
    total_num_graphs = 100
    ds = FakeDataset(num_graphs=total_num_graphs, avg_num_nodes=10)
    total_num_nodes = sum(d.num_nodes for d in ds)
    total_num_edges = sum(d.num_edges for d in ds)

    # Loader should create 10 batches of 11 graphs each (10 real + 1 padding
    # graph).
    expected_num_batches = 10
    padded_batch_size = 11
    fixed_size_options = FixedSizeOptions(num_nodes=total_num_nodes,
                                          num_graphs=padded_batch_size)
    loader = create_loader(ds,
                           batch_size=padded_batch_size,
                           fixed_size_options=fixed_size_options,
                           fixed_size_strategy=FixedSizeStrategy.StreamPack)
    batches_created = sum(1 for _ in loader)

    assert batches_created == expected_num_batches

    # Loader should create only 1 batch since there is space for all graphs
    # and one padding graph.
    expected_num_batches = 1
    fixed_size_options = FixedSizeOptions(num_nodes=total_num_nodes + 1,
                                          num_edges=total_num_edges + 1,
                                          num_graphs=101)
    loader = create_loader(ds,
                           batch_size=101,
                           fixed_size_options=fixed_size_options,
                           fixed_size_strategy=FixedSizeStrategy.StreamPack)
    batches_created = sum(1 for _ in loader)

    assert batches_created == expected_num_batches

    # There is no space for padding graph in the first batch (not enough
    # graphs) so loader should create two batches.
    expected_num_batches = 2
    fixed_size_options = FixedSizeOptions(num_nodes=total_num_nodes + 1,
                                          num_edges=total_num_edges + 1,
                                          num_graphs=100)
    loader = create_loader(ds,
                           batch_size=100,
                           fixed_size_options=fixed_size_options,
                           fixed_size_strategy=FixedSizeStrategy.StreamPack)
    batches_created = sum(1 for _ in loader)

    assert batches_created == expected_num_batches

    # There is no space for padding graph in the first batch (not enough
    # nodes) so loader should create two batches.
    expected_num_batches = 2
    fixed_size_options = FixedSizeOptions(num_nodes=total_num_nodes,
                                          num_edges=total_num_edges + 1,
                                          num_graphs=101)
    loader = create_loader(ds,
                           batch_size=101,
                           fixed_size_options=fixed_size_options,
                           fixed_size_strategy=FixedSizeStrategy.StreamPack)
    batches_created = sum(1 for _ in loader)

    assert batches_created == expected_num_batches

    # There is no space for padding graph in the first batch (not enough
    # edges) so loader should create two batches.
    expected_num_batches = 2
    fixed_size_options = FixedSizeOptions(num_nodes=total_num_nodes + 1,
                                          num_edges=total_num_edges,
                                          num_graphs=101)
    loader = create_loader(ds,
                           batch_size=101,
                           fixed_size_options=fixed_size_options,
                           fixed_size_strategy=FixedSizeStrategy.StreamPack)
    batches_created = sum(1 for _ in loader)

    assert batches_created == expected_num_batches


def test_fixed_size_dataloader_with_default_values(fake_large_dataset):
    ds = fake_large_dataset
    batch_size = 10
    padded_batch_size = batch_size + 1
    # The default value of `num_nodes` should be large enough so it's possible
    # to always pick 10 graphs and create additional padding graph.
    loader = FixedSizeDataLoader(ds, batch_size=padded_batch_size)
    expected_batches = 10

    num_batches = sum(1 for _ in loader)
    assert expected_batches == num_batches

    # DataLoader should correctly capture the number of nodes from sampler.
    sampler = StreamPackingSampler(ds, max_num_graphs=batch_size)
    loader = FixedSizeDataLoader(ds,
                                 batch_size=padded_batch_size,
                                 batch_sampler=sampler)

    num_batches = 0
    for batch in loader:
        assert batch.num_nodes == sampler.max_num_nodes + 1
        num_batches += 1
    assert expected_batches == num_batches


@pytest.mark.parametrize('create_loader',
                         [FixedSizeDataLoader, IPUFixedSizeDataLoader])
def test_fixed_size_dataloader_with_custom_batch_sampler(create_loader):
    total_num_graphs = 20
    batch_size = 5
    ds = FakeDataset(num_graphs=total_num_graphs, avg_num_nodes=10)

    class DummySampler:
        def __init__(self, data_source, batch_size):
            self.data_source = data_source
            self.batch_size = batch_size

        def __iter__(self):
            for _ in range(len(self)):
                yield [0] * self.batch_size

        def __len__(self):
            return len(self.data_source) // self.batch_size

    sampler = DummySampler(ds, batch_size - 1)

    with pytest.raises(ValueError):
        loader = create_loader(
            ds,
            batch_size=5,
            batch_sampler=sampler,
            fixed_size_strategy=FixedSizeStrategy.StreamPack)

    loader = FixedSizeDataLoader(ds,
                                 batch_size=batch_size,
                                 batch_sampler=sampler)

    num_batches = sum(1 for _ in loader)
    assert num_batches == 5
