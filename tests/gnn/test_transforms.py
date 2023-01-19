# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import copy
import numbers

import pytest
import torch
from torch_geometric.datasets import KarateClub

from poppyg import Pad


def _hetero_edges_gen(data):
    for type, type_values in zip(data.edge_types, data.edge_stores):
        for key, val in type_values.items():
            yield type, key, val


def _hetero_nodes_gen(data, include_keys, custom_cond=None):
    if not custom_cond:
        cond = (lambda _: True) if include_keys is None else (
            lambda key: key in include_keys)
    else:
        cond = custom_cond
    for type, type_values in zip(data.node_types, data.node_stores):
        for key, val in type_values.items():
            if cond(key):
                yield type, key, val


def _check_hetero_data_nodes(original, padded, max_num_nodes, node_pad_value,
                             include_keys):
    node_pad_value = 0.0 if node_pad_value is None else node_pad_value
    node_pad_value_is_number = isinstance(node_pad_value, numbers.Number)
    pad_value = node_pad_value
    for type, feature_key, feature_val in _hetero_nodes_gen(
            padded, include_keys):
        # Check the number of nodes.
        if isinstance(max_num_nodes, dict):
            if type in max_num_nodes:
                assert feature_val.shape[0] == max_num_nodes[type]
        else:
            assert feature_val.shape[0] == max_num_nodes

        compare_pad_start_idx = original[type][feature_key].shape[0]

        # Check padded area.
        if not node_pad_value_is_number:
            pad_value = node_pad_value[type] if type in node_pad_value else 0.0
        assert all(i == pad_value
                   for i in torch.flatten(feature_val[compare_pad_start_idx:]))
        # Compare non-padded area with the original.
        assert torch.equal(original[type][feature_key],
                           feature_val[:compare_pad_start_idx])
    if include_keys:
        # Check all the nodes not included in include_keys.
        all_except_include_keys_cond = (lambda key: key not in include_keys)
        for type, feature_key, feature_val in _hetero_nodes_gen(
                padded, include_keys, all_except_include_keys_cond):
            assert torch.equal(original[type][feature_key], feature_val)


def _check_hetero_data_edges(original, padded, max_num_edges, edge_pad_value,
                             include_keys):

    for type, feature_key, feature_val in _hetero_edges_gen(padded):
        compare_pad_start_idx = original[type].num_edges

        padding_added = not include_keys or 'x' in include_keys
        if padding_added:
            src_node, _, dst_node = type
            expected_num_edges = None
            if max_num_edges:
                if isinstance(max_num_edges, numbers.Number):
                    expected_num_edges = max_num_edges
                elif type in max_num_edges.keys():
                    expected_num_edges = max_num_edges[type]
                else:
                    padding_added = False
            elif original[src_node].x.shape == padded[
                    src_node].x.shape or original[dst_node].x.shape == padded[
                        dst_node].x.shape:
                # The src or dst node of the edge was not padded so neither
                # was the edge.
                padding_added = False

        if feature_key == 'edge_index':
            if padding_added:
                # Check padded area size.
                if not expected_num_edges:
                    padded_src_nodes = padded[src_node].num_nodes
                    padded_dst_nodes = padded[dst_node].num_nodes
                    expected_num_edges = padded_src_nodes * padded_dst_nodes

                # Check the number of edges.
                assert padded[type][feature_key].shape[1] == expected_num_edges

                # Check padded area values.
                src_nodes = original[type[0]].num_nodes
                assert all(i == src_nodes for i in torch.flatten(
                    feature_val[0, compare_pad_start_idx:]))
                dst_nodes = original[type[2]].num_nodes
                assert all(i == dst_nodes for i in torch.flatten(
                    feature_val[1, compare_pad_start_idx:]))

            # Compare non-padded area with the original.
            assert torch.equal(original[type][feature_key],
                               feature_val[:, :compare_pad_start_idx])
        else:
            if padding_added:
                # Check padded area size.
                if not expected_num_edges:
                    expected_num_edges = original[type].num_edges
                assert padded[type][feature_key].shape[0] == expected_num_edges

                # Check padded area values.
                compare_val = 0.0 if edge_pad_value is None else edge_pad_value
                assert all(i == compare_val for i in torch.flatten(
                    feature_val[compare_pad_start_idx:, :]))

            # Compare non-padded area with the original.
            assert torch.equal(original[type][feature_key],
                               feature_val[:compare_pad_start_idx, :])


def _check_hetero_data(original,
                       padded,
                       max_num_nodes,
                       max_num_edges=None,
                       node_pad_value=None,
                       edge_pad_value=None,
                       include_keys=None):
    _check_hetero_data_nodes(original, padded, max_num_nodes, node_pad_value,
                             include_keys)
    _check_hetero_data_edges(original, padded, max_num_edges, edge_pad_value,
                             include_keys)


def _homo_node_gen(data):
    for k in data.get_all_tensor_attrs():
        yield k.attr_name, data[k.attr_name]


def _check_homo_data_nodes(original, padded, max_num_nodes, node_pad_value):
    node_pad_value = 0.0 if node_pad_value is None else node_pad_value
    node_pad_value_is_number = isinstance(node_pad_value, numbers.Number)
    pad_value = node_pad_value
    for feature_key, feature_val in _homo_node_gen(padded):
        # Check the number of nodes.
        assert feature_val.shape[0] == max_num_nodes

        compare_pad_start_idx = original[feature_key].shape[0]

        # Check padded area.
        if not node_pad_value_is_number:
            pad_value = node_pad_value[
                feature_key] if feature_key in node_pad_value else 0.0
        assert all(i == pad_value
                   for i in torch.flatten(feature_val[compare_pad_start_idx:]))

        # Compare non-padded area with the original.
        assert torch.equal(original[feature_key],
                           feature_val[:compare_pad_start_idx])


def _check_homo_data_edges(original, padded, max_num_edges, edge_pad_value):
    # Check edge_index:

    # Check the number of edges.
    if max_num_edges is not None:
        assert padded.edge_index.shape[1] == max_num_edges

    # Check padded area values.
    compare_pad_start_idx = original.num_edges
    src_nodes = original.num_nodes
    assert all(
        i == src_nodes
        for i in torch.flatten(padded.edge_index[0, compare_pad_start_idx:]))
    dst_nodes = original.num_nodes
    assert all(
        i == dst_nodes
        for i in torch.flatten(padded.edge_index[1, compare_pad_start_idx:]))

    # Compare non-padded area with the original.
    assert torch.equal(original.edge_index,
                       padded.edge_index[:, :compare_pad_start_idx])

    # Check edge_attr:

    # Check padded area size.
    if max_num_edges is not None:
        assert padded.edge_attr.shape[0] == max_num_edges

    # Check padded area values.
    compare_val = 0.0 if edge_pad_value is None else edge_pad_value
    assert all(
        i == compare_val
        for i in torch.flatten(padded.edge_attr[compare_pad_start_idx:, :]))

    # Compare non-padded area with the original.
    assert torch.equal(original.edge_attr,
                       padded.edge_attr[:compare_pad_start_idx, :])


def _check_homo_data(original,
                     padded,
                     max_num_nodes,
                     max_num_edges=None,
                     node_pad_value=None,
                     edge_pad_value=None):
    _check_homo_data_nodes(original, padded, max_num_nodes, node_pad_value)
    _check_homo_data_edges(original, padded, max_num_edges, edge_pad_value)


def test_pad_repr():
    pad_str = 'Pad(max_num_nodes=10, max_num_edges=15, '\
              'node_pad_value=3.0, edge_pad_value=1.5)'
    assert eval(pad_str).__repr__() == pad_str  # pylint: disable=eval-used


@pytest.mark.parametrize('node_pad_value',
                         [None, 123, {
                             'x': 12
                         }, {
                             'x': 12,
                             'z': 34,
                             'pos': 56
                         }])
@pytest.mark.parametrize('edge_pad_value', [None, 321])
def test_pad_value(molecule, node_pad_value, edge_pad_value):
    original = molecule
    # Pad does override it's input, so we need to deepcopy the origin.
    padded = copy.deepcopy(original)

    max_num_nodes = 32
    max_num_edges = max_num_nodes**2
    pad_params = {'max_num_nodes': max_num_nodes}
    if node_pad_value:
        pad_params['node_pad_value'] = node_pad_value
    if edge_pad_value:
        pad_params['edge_pad_value'] = edge_pad_value
    transform = Pad(**pad_params)
    padded = transform(padded)

    assert padded.num_nodes == max_num_nodes
    assert padded.num_edges == max_num_edges

    _check_homo_data(original, padded, max_num_nodes, max_num_edges,
                     node_pad_value, edge_pad_value)


@pytest.mark.parametrize('max_num_nodes',
                         [None, {
                             'v0': 100
                         }, {
                             'v0': 100,
                             'v1': 100
                         }])
@pytest.mark.parametrize('node_pad_value', [None, 200.0, {'x': 200}])
@pytest.mark.parametrize('edge_pad_value', [None, 121.0])
@pytest.mark.parametrize('max_num_edges',
                         [None, 600, {
                             ('v0', 'e0', 'v1'): 1000
                         }])
def test_pad_hetero_data(fake_hetero_dataset, max_num_nodes, node_pad_value,
                         edge_pad_value, max_num_edges):
    original = fake_hetero_dataset
    # Pad does override it's input, so we need to deepcopy the origin.
    padded = copy.deepcopy(original)

    if max_num_nodes is None:
        or_num_nodes = original.num_nodes
        max_num_nodes = or_num_nodes + 10

    pad_params = {'max_num_nodes': max_num_nodes}
    if node_pad_value is not None:
        pad_params['node_pad_value'] = node_pad_value
    if edge_pad_value is not None:
        pad_params['edge_pad_value'] = edge_pad_value
    if max_num_edges is not None:
        pad_params['max_num_edges'] = max_num_edges

    transform = Pad(**pad_params)
    padded = transform(padded)

    _check_hetero_data(original, padded, max_num_nodes, max_num_edges,
                       node_pad_value, edge_pad_value)


@pytest.mark.parametrize('node_pad_value',
                         [None, 123, {
                             'x': 12
                         }, {
                             'x': 12,
                             'z': 34
                         }])
@pytest.mark.parametrize('include_keys',
                         [None, 'z', 'x', 'non_existing_attr', ('x', 'z')])
def test_pad_include_keys(molecule, node_pad_value, include_keys):
    original = molecule
    # Pad does override it's input, so we need to deepcopy the origin.
    padded = copy.deepcopy(original)

    max_num_nodes = 32
    max_num_edges = max_num_nodes**2
    pad_params = {'max_num_nodes': max_num_nodes, 'include_keys': include_keys}
    if node_pad_value:
        pad_params['node_pad_value'] = node_pad_value
    transform = Pad(**pad_params)
    padded = transform(padded)

    expected_num_edges = None if include_keys else max_num_edges
    _check_homo_data(original, padded, max_num_nodes, expected_num_edges,
                     node_pad_value)


@pytest.mark.parametrize('include_keys',
                         [None, 'y', 'x', 'non_existing_attr', ('x', 'y')])
def test_pad_include_keys_hetero_data(fake_hetero_dataset, include_keys):
    original = fake_hetero_dataset
    # Pad does override it's input, so we need to deepcopy the origin.
    padded = copy.deepcopy(original)

    or_num_nodes = original.num_nodes
    padded_num_nodes = or_num_nodes + 10
    transform = Pad(max_num_nodes=padded_num_nodes, include_keys=include_keys)
    padded = transform(padded)

    _check_hetero_data(original,
                       padded,
                       padded_num_nodes,
                       include_keys=include_keys)


def test_pad_invalid_max_num_nodes(molecule):
    transform = Pad(max_num_nodes=3)

    with pytest.raises(AssertionError, match='Too many nodes'):
        transform(molecule)


def test_pad_invalid_max_num_nodes_hetero_data(fake_hetero_dataset):
    transform = Pad(max_num_nodes=3)

    with pytest.raises(AssertionError, match='Too many nodes'):
        transform(fake_hetero_dataset)


def test_pad_invalid_max_num_edges(molecule):
    transform = Pad(max_num_nodes=32, max_num_edges=10)

    with pytest.raises(AssertionError, match='Too many edges'):
        transform(molecule)


def test_pad_invalid_max_num_edges_hetero_data(fake_hetero_dataset):
    transform = Pad(max_num_nodes=132, max_num_edges=10)

    with pytest.raises(AssertionError, match='Too many edges'):
        transform(fake_hetero_dataset)


testdata_mask_values = [True, False]
testdata_mask_params = ['train_mask', 'test_mask']


@pytest.mark.parametrize('mask_pad_value', testdata_mask_values)
@pytest.mark.parametrize('mask_param', testdata_mask_params)
def test_pad_node_additional_attr_mask(molecule, mask_param, mask_pad_value):
    mask = torch.randn(molecule.num_nodes) > 0
    setattr(molecule, mask_param, mask)
    padding_num = 20

    max_num_nodes = int(molecule.num_nodes) + padding_num
    max_num_edges = molecule.num_edges + padding_num

    params = {f'{mask_param}_pad_value': mask_pad_value}
    transform = Pad(max_num_nodes, max_num_edges, node_pad_value=0.1, **params)
    padded = transform(molecule)
    padded_mask = getattr(padded, mask_param)

    assert padded_mask.ndim == 1
    assert padded_mask.size()[0] == max_num_nodes
    assert torch.all(padded_mask[-padding_num:] == mask_pad_value)


@pytest.mark.parametrize('mask_pad_value', testdata_mask_values)
def test_pad_node_with_train_mask(mask_pad_value):
    data = KarateClub()[0]
    padding_num = 23

    max_num_nodes = int(data.num_nodes) + padding_num
    max_num_edges = data.num_edges + padding_num

    transform = Pad(max_num_nodes,
                    max_num_edges,
                    node_pad_value=0.1,
                    train_mask_pad_value=mask_pad_value)
    padded = transform(data)
    padded_mask = padded.train_mask

    assert padded_mask.ndim == 1
    assert padded_mask.size()[0] == max_num_nodes
    assert torch.all(padded_mask[-padding_num:] == mask_pad_value)


@pytest.mark.parametrize('apply_transform', [True, False])
def test_pad_perf(pyg_qm9, benchmark, apply_transform):
    num_graphs = 10000
    dataset = pyg_qm9[0:num_graphs]

    if apply_transform:
        include_keys = ('z', 'edge_attr', 'edge_index', 'batch', 'y')
        dataset.transform = Pad(max_num_nodes=32, include_keys=include_keys)

    def loop():
        for _ in dataset:
            pass

    benchmark(loop)
