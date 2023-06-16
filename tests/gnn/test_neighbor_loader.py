# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.testing import (
    get_random_edge_index,
    onlyNeighborSampler,
)
from torch_geometric.utils import (
    is_undirected, )

from poptorch_geometric.fixed_size_options import FixedSizeOptions
from poptorch_geometric.neighbor_loader import FixedSizeNeighborLoader


def validate_fixed_data_format(loader: FixedSizeNeighborLoader,
                               fixed_size_options: FixedSizeOptions,
                               is_hetero_data: bool,
                               debug_print: bool = False):

    for index in range(0, len(loader), loader.batch_size):  # pylint: disable=too-many-nested-blocks

        indices = list(range(index, index + loader.batch_size))
        dynamic = loader.nativeCollate([indices])
        fixed = loader.fixedSizeCollate(dynamic)

        if is_hetero_data:
            dynamic_dict_store = {
                '_node_store_dict': dynamic.__dict__['_node_store_dict'],
                '_edge_store_dict': dynamic.__dict__['_edge_store_dict']
            }
            fixed_dict_store = {
                '_node_store_dict': fixed.__dict__['_node_store_dict'],
                '_edge_store_dict': fixed.__dict__['_edge_store_dict']
            }
        else:
            dynamic_dict_store = {
                '_store': {
                    "Data:": dynamic.__dict__['_store'].__dict__['_mapping']
                }
            }
            fixed_dict_store = {
                '_store': {
                    "Data:": fixed.__dict__['_store'].__dict__['_mapping']
                }
            }

        for storage_type in dynamic_dict_store:
            if debug_print:
                print(f"Store [{storage_type}]")

            if storage_type == '_edge_store_dict':
                pad_value = fixed_size_options.edge_pad_value
            else:
                pad_value = fixed_size_options.node_pad_value

            dynamic_dict_group = dynamic_dict_store[storage_type]
            fixed_dict_group = fixed_dict_store[storage_type]

            for group in dynamic_dict_group:
                if debug_print:
                    print(f"Group [{group}]")

                dynamic_dict = dynamic_dict_group[group]
                fixed_dict = fixed_dict_group[group]

                # check if values are padded as expected
                for key in dynamic_dict:

                    # Batch size is used only for sampling
                    if key == 'batch_size':
                        continue

                    dynamic_tensor = dynamic_dict[key]
                    fixed_tensor = fixed_dict[key]

                    if debug_print:
                        print(f"Key: [{key}]")
                        print("Dynamic:", dynamic_tensor)
                        print("Fixed  :", fixed_tensor)

                    if dynamic_tensor.dim() < 2:
                        dynamic_tensor = [dynamic_tensor]
                        fixed_tensor = [fixed_tensor]

                    for i in range(0, len(dynamic_tensor)):  # pylint: disable=consider-using-enumerate
                        dynamic_dim = dynamic_tensor[i]
                        fixed_dim = fixed_tensor[i]
                        valid_range = range(
                            0, min(len(dynamic_dim), len(fixed_dim)))
                        fixed_range = range(len(valid_range), len(fixed_dim))

                        for j in valid_range:
                            assert dynamic_dim[j] == fixed_dim[j]

                        # Dummy (padded) edge_index should point to dummy node
                        if key == 'edge_index':
                            if is_hetero_data:
                                n_id_tensor = fixed_dict_store[  # pylint: disable=line-too-long
                                    '_node_store_dict'][
                                        group[0 if i < 1 else -1]]['n_id']
                            else:
                                assert fixed_size_options.num_edges == len(
                                    fixed_dim), f"Incorrect padding for {key}"
                                n_id_tensor = fixed_dict['n_id']
                            for j in fixed_range:
                                assert n_id_tensor[fixed_dim[j]] == pad_value
                        # Dummy (padded) value check
                        else:
                            for j in fixed_range:
                                assert fixed_dim[j] == pad_value


def is_subset(subedge_index, edge_index, src_idx, dst_idx):
    num_nodes = int(edge_index.max()) + 1
    idx = num_nodes * edge_index[0] + edge_index[1]
    subidx = num_nodes * src_idx[subedge_index[0]] + dst_idx[subedge_index[1]]
    mask = torch.from_numpy(np.isin(subidx, idx))
    return int(mask.sum()) == mask.numel()


@onlyNeighborSampler
@pytest.mark.parametrize('subgraph_type', list(SubgraphType))
def test_homo_neighbor_loader_basic(subgraph_type):

    torch.manual_seed(12345)

    data = Data()

    data.x = torch.arange(15)
    data.edge_index = get_random_edge_index(15, 15, 75, torch.int64)
    data.edge_attr = torch.arange(75)
    use_batch_size = 5

    default_loader = NeighborLoader(
        data,
        num_neighbors=[5] * 2,
        batch_size=use_batch_size,
        subgraph_type=subgraph_type,
    )

    fixed_size_options = FixedSizeOptions.from_loader(default_loader)

    loader = FixedSizeNeighborLoader(
        data,
        num_neighbors=[5] * 2,
        batch_size=use_batch_size,
        subgraph_type=subgraph_type,
        fixed_size_options=fixed_size_options,
    )

    validate_fixed_data_format(loader=loader,
                               fixed_size_options=fixed_size_options,
                               is_hetero_data=False)

    assert len(loader) == len(data.x) // use_batch_size

    batch = next(iter(loader))

    assert isinstance(batch, Data)
    assert batch.n_id[:1].tolist() == [0]

    for i, batch in enumerate(loader):
        assert isinstance(batch, Data)
        assert batch.x.size(0) <= 101
        assert batch.n_id.size() == (batch.num_nodes, )
        assert batch.x.min() >= 0 and batch.x.max() < 101
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes

        # Input nodes are always sampled first:
        assert torch.equal(
            batch.x[:use_batch_size],
            torch.arange(i * use_batch_size, (i + 1) * use_batch_size))

        if subgraph_type != SubgraphType.bidirectional:
            assert batch.edge_attr.min() >= 0
            assert batch.edge_attr.max() < 500

            assert is_subset(
                batch.edge_index.to(torch.int64),
                data.edge_index.to(torch.int64),
                batch.x,
                batch.x,
            )


@onlyNeighborSampler
@pytest.mark.parametrize('subgraph_type', list(SubgraphType))
def test_hetero_neighbor_loader_basic(subgraph_type):
    dtype = torch.int64

    torch.manual_seed(12345)

    data = HeteroData()

    data['paper'].x = torch.arange(15)
    data['author'].x = torch.arange(15, 45)

    edge_index = get_random_edge_index(15, 15, 45, dtype)
    data['paper', 'paper'].edge_index = edge_index
    data['paper', 'paper'].edge_attr = torch.arange(45)
    edge_index = get_random_edge_index(15, 30, 90, dtype)
    data['paper', 'author'].edge_index = edge_index
    data['paper', 'author'].edge_attr = torch.arange(45, 135)
    edge_index = get_random_edge_index(30, 15, 150, dtype)
    data['author', 'paper'].edge_index = edge_index
    data['author', 'paper'].edge_attr = torch.arange(200, 250)

    batch_size = 2

    with pytest.raises(ValueError, match="hops must be the same across all"):
        default_loader = NeighborLoader(
            data,
            num_neighbors={
                ('paper', 'to', 'paper'): [-1],
                ('paper', 'to', 'author'): [-1, -1],
                ('author', 'to', 'paper'): [-1, -1],
            },
            input_nodes='paper',
            batch_size=batch_size,
            subgraph_type=subgraph_type,
        )

        fixed_size_options = FixedSizeOptions.from_loader(default_loader)

        loader = FixedSizeNeighborLoader(
            data,
            num_neighbors={
                ('paper', 'to', 'paper'): [-1],
                ('paper', 'to', 'author'): [-1, -1],
                ('author', 'to', 'paper'): [-1, -1],
            },
            input_nodes='paper',
            batch_size=batch_size,
            subgraph_type=subgraph_type,
            fixed_size_options=fixed_size_options,
        )
        next(iter(loader))

    default_loader = NeighborLoader(
        data,
        num_neighbors=[10] * 2,
        input_nodes='paper',
        batch_size=batch_size,
        subgraph_type=subgraph_type,
    )

    fixed_size_options = FixedSizeOptions.from_loader(default_loader)

    loader = FixedSizeNeighborLoader(data,
                                     num_neighbors=[10] * 2,
                                     input_nodes='paper',
                                     batch_size=batch_size,
                                     subgraph_type=subgraph_type,
                                     fixed_size_options=fixed_size_options)
    assert len(loader) > 0

    validate_fixed_data_format(loader=loader,
                               fixed_size_options=fixed_size_options,
                               is_hetero_data=True)


@onlyNeighborSampler
@pytest.mark.parametrize('subgraph_type', list(SubgraphType))
def test_hetero_neighbor_loader_large(subgraph_type):
    dtype = torch.int64

    torch.manual_seed(12345)

    data = HeteroData()

    data['paper'].x = torch.arange(20)
    data['author'].x = torch.arange(20, 220)

    edge_index = get_random_edge_index(20, 20, 40, dtype)
    data['paper', 'paper'].edge_index = edge_index
    data['paper', 'paper'].edge_attr = torch.arange(40)
    edge_index = get_random_edge_index(20, 50, 250, dtype)
    data['paper', 'author'].edge_index = edge_index
    data['paper', 'author'].edge_attr = torch.arange(40, 300)
    edge_index = get_random_edge_index(50, 20, 250, dtype)
    data['author', 'paper'].edge_index = edge_index
    data['author', 'paper'].edge_attr = torch.arange(300, 400)

    batch_size = 2

    with pytest.raises(ValueError, match="hops must be the same across all"):
        default_loader = NeighborLoader(
            data,
            num_neighbors={
                ('paper', 'to', 'paper'): [-1],
                ('paper', 'to', 'author'): [-1, -1],
                ('author', 'to', 'paper'): [-1, -1],
            },
            input_nodes='paper',
            batch_size=batch_size,
            subgraph_type=subgraph_type,
        )

        fixed_size_options = FixedSizeOptions.from_loader(default_loader)

        loader = FixedSizeNeighborLoader(
            data,
            num_neighbors={
                ('paper', 'to', 'paper'): [-1],
                ('paper', 'to', 'author'): [-1, -1],
                ('author', 'to', 'paper'): [-1, -1],
            },
            input_nodes='paper',
            batch_size=batch_size,
            subgraph_type=subgraph_type,
            fixed_size_options=fixed_size_options,
        )
        next(iter(loader))

    default_loader = NeighborLoader(
        data,
        num_neighbors=[10] * 2,
        input_nodes='paper',
        batch_size=batch_size,
        subgraph_type=subgraph_type,
    )

    fixed_size_options = FixedSizeOptions.from_loader(default_loader)

    loader = FixedSizeNeighborLoader(data,
                                     num_neighbors=[10] * 2,
                                     input_nodes='paper',
                                     batch_size=batch_size,
                                     subgraph_type=subgraph_type,
                                     add_pad_masks=True,
                                     fixed_size_options=fixed_size_options)
    assert len(loader) > 0

    validate_fixed_data_format(loader=loader,
                               fixed_size_options=fixed_size_options,
                               is_hetero_data=True)

    for batch in loader:
        assert isinstance(batch, HeteroData)

        # Test node type selection:
        assert set(batch.node_types) == {'paper', 'author'}

        assert batch['paper'].n_id.size() == (batch['paper'].num_nodes, )
        assert batch['paper'].x.size(0) <= 20 + 1
        assert batch['paper'].x.min() >= 0 and batch['paper'].x.max() < 40 + 1

        assert batch['author'].n_id.size() == (batch['author'].num_nodes, )
        assert batch['author'].x.size(0) <= 50
        assert batch['author'].x.max() < 220

        # Test edge type selection:
        assert set(batch.edge_types) == {('paper', 'to', 'paper'),
                                         ('paper', 'to', 'author'),
                                         ('author', 'to', 'paper')}

        row, col = batch['paper', 'paper'].edge_index
        assert row.min() >= 0 and row.max() < batch['paper'].num_nodes
        assert col.min() >= 0 and col.max() < batch['paper'].num_nodes

        if subgraph_type != SubgraphType.bidirectional:
            assert batch['paper', 'paper'].e_id.size() == (row.numel(), )
            value = batch['paper', 'paper'].edge_attr
            assert value.min() >= 0 and value.max() < 40

            assert is_subset(
                batch['paper', 'paper'].edge_index.to(
                    torch.int64)[:, batch['paper', 'paper'].edges_mask],
                data['paper', 'paper'].edge_index.to(torch.int64),
                batch['paper'].x,
                batch['paper'].x,
            )
        elif subgraph_type != SubgraphType.directional:
            assert 'e_id' not in batch['paper', 'paper']  # pylint: disable=no-value-for-parameter
            assert 'edge_attr' not in batch['paper', 'paper']  # pylint: disable=no-value-for-parameter

            assert is_undirected(batch['paper', 'paper'].edge_index)  # pylint: disable=no-value-for-parameter

        row, col = batch['paper', 'author'].edge_index
        assert row.min() >= 0 and row.max() < batch['paper'].num_nodes
        assert col.min() >= 0 and col.max() < batch['author'].num_nodes
