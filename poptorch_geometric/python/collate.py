# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from functools import singledispatch
try:
    from functools import singledispatchmethod
except ImportError:
    from singledispatchmethod import singledispatchmethod
from itertools import chain

import torch
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.data.data import BaseData
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.transforms import Pad

from poptorch_geometric.pyg_collate import Collater
from poptorch_geometric.utils import DataBatch, HeteroDataBatch

from poptorch._utils import combine_batch_tensors_gen

from . import types

__all__ = ['FixedSizeCollater', 'CombinedBatchingCollater']


def make_exclude_keys(include_keys: Union[List[str], Tuple[str, ...]],
                      data: BaseData) -> Tuple[str, ...]:
    return tuple(set(data.keys) - set(include_keys))


def _divide_evenly_formula(amount: int, pieces: int) -> List[int]:
    minimum = amount // pieces
    extra = amount - minimum * pieces
    return [minimum + (1 if i < extra else 0) for i in range(pieces)]


@singledispatch
def _divide_evenly(data, num_pad_graphs, num_pad_nodes, num_pad_edges):  # pylint: disable=unused-argument
    raise ValueError(f'Unsupported data type: {type(data)}')


@_divide_evenly.register(Data)
def _(_, num_pad_graphs: int, num_pad_nodes: int,
      num_pad_edges: int) -> Tuple[List[int], List[int]]:
    return _divide_evenly_formula(num_pad_nodes,
                                  num_pad_graphs), _divide_evenly_formula(
                                      num_pad_edges, num_pad_graphs)


@_divide_evenly.register(HeteroData)
def _(_, num_pad_graphs: int, num_pad_nodes: Dict[NodeType, int],
      num_pad_edges: Dict[EdgeType, int]
      ) -> Tuple[List[Dict[NodeType, int]], List[Dict[EdgeType, int]]]:
    def calc_pads(num_pad_elems):
        pad_elems = [dict() for i in range(num_pad_graphs)]
        for type_, pad_val in num_pad_elems.items():
            pad_per_graph = _divide_evenly_formula(pad_val, num_pad_graphs)
            for graph_idx, graph_pad in enumerate(pad_per_graph):
                pad_elems[graph_idx][type_] = graph_pad
        return pad_elems

    pad_nodes = calc_pads(num_pad_nodes)
    pad_edges = calc_pads(num_pad_edges)
    return pad_nodes, pad_edges


@singledispatch
def _generate_data_to_pad(data_to_pad_dict):
    raise ValueError(f'Unsupported data type: {type(data_to_pad_dict)}')


@_generate_data_to_pad.register(Data)
def _(data_to_pad_dict: dict) -> Data:
    return Data.from_dict(data_to_pad_dict)


@_generate_data_to_pad.register(HeteroData)
def _(data_to_pad_dict: dict) -> HeteroData:
    return HeteroData(data_to_pad_dict)


def _reset_dim(shape: torch.Size, key: str = None) -> List[int]:
    shape = list(shape)
    if len(shape) > 1:
        shape[1 if key == 'edge_index' else 0] = 0
    else:
        return list([0])
    return shape


def _reset_attr(value: Any, key: str = None) -> Any:
    """Reset value to the default of its type. In case of torch.Tensor, it
    returns a tensor with one of the dims set to 0. The dim is
    determined based on the key.
    """
    if isinstance(value, torch.Tensor):
        # NOTE: It has to be torch.zeros - creating a Tensor directly
        # (through torch.tensor) with 0 in shape ends up in creating a
        # tensor with wrong dimensions.
        return torch.zeros(_reset_dim(value.shape, key), dtype=value.dtype)
    return type(value)()


def _make_node_slices_gen(data_list: List[BaseData]) -> List[slice]:
    data_type = type(data_list[0])
    return list(_data_slice_gen.dispatch(data_type)(data_list, 'num_nodes'))


def _make_data_edge_slice_gen(data_list: List[BaseData]) -> List[slice]:
    data_type = type(data_list[0])
    return list(_data_slice_gen.dispatch(data_type)(data_list, 'num_edges'))


@singledispatch
def _data_slice_gen(data_list, attr) -> Generator[slice, None, None]:
    raise ValueError(f'Unsupported data type: {type(data_list[0])}')


@_data_slice_gen.register(Data)
def _(data_list: List[Data], attr: str) -> Generator[slice, None, None]:
    start = 0
    end = 0
    for data in data_list:
        end += getattr(data, attr)
        yield slice(start, end)
        start = end


@_data_slice_gen.register(HeteroData)
def _(data_list: List[HeteroData], attr: str) -> Generator[slice, None, None]:
    start = 0
    end = 0
    is_nodes_gen = (attr == 'num_nodes')
    for data in data_list:
        for node_type in (data.node_stores
                          if is_nodes_gen else data.edge_stores):
            end += getattr(node_type, attr)
            yield slice(start, end)
            start = end


def _create_preserve_mask(num_elems: int, num_elems_to_trim: int,
                          slices: List[slice]) -> List[bool]:
    # Prevent deletion of all elements from a single graph.
    removable_nodes_mask = torch.ones(num_elems, dtype=torch.bool)
    for data_slice in slices:
        if data_slice.start < data_slice.stop:
            mask_slice = removable_nodes_mask[data_slice]
            mask_slice[torch.randint(high=len(mask_slice), size=(1, ))] = False
    indices = torch.arange(0, num_elems)[removable_nodes_mask]

    # Randomly select elements to remove.
    prune_indices = indices[torch.randperm(
        len(indices))][:num_elems_to_trim].type(torch.long)
    preserve_mask = torch.ones(num_elems, dtype=torch.bool)
    preserve_mask[prune_indices] = False

    return preserve_mask


@singledispatch
def _prune_data_nodes(data_list, preserve_nodes_mask, node_slices):  # pylint: disable=unused-argument
    raise ValueError(f'Unsupported data types: {type(data_list[0])}')


@_prune_data_nodes.register(Data)
def _(data_list: List[Data], preserve_nodes_mask: torch.Tensor,
      node_slices: List[slice]) -> List[Data]:
    return [
        data.subgraph(preserve_nodes_mask[slice])
        for data, slice in zip(data_list, node_slices)
    ]


@_prune_data_nodes.register(HeteroData)
def _(data_list: List[HeteroData], preserve_nodes_mask: torch.Tensor,
      node_slices: List[slice]) -> List[HeteroData]:
    node_slices_iter = iter(node_slices)
    return [
        data.subgraph({
            node_type: preserve_nodes_mask[next(node_slices_iter)]
            for node_type in data.node_types
        }) for data in data_list
    ]


@singledispatch
def _prune_data_edges(data_list, preserve_edges_mask, edge_slices):  # pylint: disable=unused-argument
    raise ValueError(f'Unsupported data types: {type(data_list[0])}')


@_prune_data_edges.register(Data)
def _(data_list: List[Data], preserve_edges_mask: torch.Tensor,
      edge_slices: List[slice]) -> Data:
    return [
        data.edge_subgraph(preserve_edges_mask[slc])
        for data, slc in zip(data_list, edge_slices)
    ]


@_prune_data_edges.register(HeteroData)
def _(data_list: List[HeteroData], preserve_edges_mask: torch.Tensor,
      edge_slices: List[slice]) -> HeteroData:
    edge_slices_iter = iter(edge_slices)
    return [
        data.edge_subgraph({
            edge_type: preserve_edges_mask[next(edge_slices_iter)]
            for edge_type in data.edge_types
        }) for data in data_list
    ]


@singledispatch
def _any_negative(value: int) -> bool:
    return value < 0


@_any_negative.register(dict)
def _(value: dict) -> bool:
    return any(v < 0 for v in value.values())


@singledispatch
def _all_positive(value: int) -> bool:
    return value > 0


@_all_positive.register(dict)
def _(value: dict) -> bool:
    return all(v > 0 for v in value.values())


class FixedSizeCollater(Collater):
    r"""Collates a batch of graphs as a
    :py:class:`torch_geometric.data.Batch` of fixed-size tensors.

    Calling an instance of this class adds an additional graphs with the
    necessary number of nodes and edges to pad the batch so that tensors have
    the size corresponding to the maximum numbers of graphs, nodes and edges
    specified during initialisation.

    Calling an instance of this class can result in :py:exc:`RuntimeError` if
    the number of graphs (if set), nodes or edges in the batch is larger than
    the requested limits.

    Args:
        num_nodes (int): The total number of nodes in the padded batch.
        num_edges (int, optional): The total number of edges in the padded
            batch. (default: :obj:`num_nodes * (num_nodes - 1)`)
        num_graphs (int, optional): The total number of graphs in the padded
            batch. Note that if it is not explicitly set then this class always
            increases the batch size by 1, i.e. given a batch size of 128
            graphs, it will return a batch with 129 graphs. Otherwise it will
            pad given batch to the :obj:`num_graphs` limit. (default:
            :obj:`None`)
        node_pad_value (float, optional): The fill value to use for node
            features. (default: :obj:`0.0`)
        edge_pad_value (float, optional): The fill value to use for edge
            features. (default: :obj:`0.0`)
        graph_pad_value (float, optional): The fill value to use for graph
            features. (default: :obj:`0.0`)
        add_masks_to_batch (bool, optional): If set to :obj:`True`, masks object
            are attached to batch result. They represents three levels of
            padding:

            - :obj:`graphs_mask` - graph level mask
            - :obj:`nodes_mask`  - node level mask
            - :obj:`edges_mask`  - edge level mask

            Mask objects indicates which elements in the batch are real
            (represented by :obj:`True` value) and which were added as a padding
            (represented by :obj:`False` value). (default: :obj:`False`)
        trim_nodes (bool, optional): If set to :obj:`True`, randomly prune
            nodes from batch to fulfill the condition of :obj:`num_nodes`.
            (default: :obj:`False`)
        trim_edges (bool, optional): If set to :obj:`True`, randomly prune
            edges from batch to fulfill the condition of :obj:`num_edges`.
            (default: :obj:`False`)
        pad_graph_defaults (dict, optional): The default values that
            will be assigned to the keys of type different than
            :class:`torch.Tensor` in the newly created padding graphs.
            (default: :obj:`None`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (list or tuple, optional): The keys to exclude
            from the graphs in the output batch. (default: :obj:`None`)
    """

    def __init__(
            self,
            num_nodes: int,
            num_edges: Optional[int] = None,
            num_graphs: Optional[int] = None,
            node_pad_value: Optional[float] = None,
            edge_pad_value: Optional[float] = None,
            graph_pad_value: Optional[float] = None,
            add_masks_to_batch: Optional[bool] = False,
            trim_nodes: Optional[bool] = False,
            trim_edges: Optional[bool] = False,
            pad_graph_defaults: Optional[Dict[str, Any]] = None,
            follow_batch: Optional[Union[List[str], Tuple[str, ...]]] = None,
            exclude_keys: Optional[Union[List[str], Tuple[str, ...]]] = None,
    ) -> None:
        super().__init__(follow_batch, exclude_keys)

        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        if num_edges:
            self.num_edges = num_edges
        else:
            # Assume fully connected graph.
            self.num_edges = num_nodes * (num_nodes - 1)

        self.node_pad_value = 0.0 if node_pad_value is None else node_pad_value
        self.edge_pad_value = 0.0 if edge_pad_value is None else edge_pad_value
        self.graph_pad_value = (0.0 if graph_pad_value is None else
                                graph_pad_value)

        self.add_masks_to_batch = add_masks_to_batch
        self.trim_nodes = trim_nodes
        self.trim_edges = trim_edges
        self.pad_graph_defaults = ({} if pad_graph_defaults is None else
                                   pad_graph_defaults)
        self.labels_type = None

    class LabelsType(Enum):
        GRAPH_LVL = 0
        NODE_LVL = 1

    def __call__(self, data_list: List[BaseData]) -> Batch:
        if not isinstance(data_list, list):
            raise TypeError(f'Expected list, got {type(data_list).__name__}.')

        if isinstance(data_list[0], Data) and hasattr(data_list[0], 'y'):
            y0_equal_num_nodes = all(data.y.shape[0] == data.num_nodes
                                     for data in data_list)
            y0_equal_ones = all(data.y.shape[0] == 1 for data in data_list)

            if y0_equal_num_nodes and not y0_equal_ones:
                self.labels_type = self.LabelsType.NODE_LVL
            elif y0_equal_ones and not y0_equal_num_nodes:
                self.labels_type = self.LabelsType.GRAPH_LVL
            else:
                assert False, "Incorrect input data. Labels `y` have" \
                              "uncompatible shapes!"

        num_real_graphs = len(data_list)
        num_pad_graphs = 1 if self.num_graphs is None \
            else self.num_graphs - num_real_graphs
        num_all_graphs = num_real_graphs + num_pad_graphs
        num_real_nodes, num_pad_nodes, num_real_edges, num_pad_edges = \
            self._calc_pad_limits(data_list)
        if self.trim_nodes and _any_negative(num_pad_nodes):
            data_list = self._prune_nodes(data_list)
            num_real_nodes, num_pad_nodes, num_real_edges, num_pad_edges = \
                self._calc_pad_limits(data_list)

        if self.trim_edges and _any_negative(num_pad_edges):
            data_list = self._prune_edges(data_list)
            num_real_nodes, num_pad_nodes, num_real_edges, num_pad_edges = \
                self._calc_pad_limits(data_list)

        if num_pad_graphs < 0 or _any_negative(num_pad_edges) or _any_negative(
                num_pad_nodes):
            raise RuntimeError('Graphs in the batch are too large. Requested '
                               f'{num_all_graphs} graphs, but batch has '
                               f'{num_real_graphs} graphs. Requested '
                               f'{self.num_nodes} nodes, but batch has '
                               f'{num_real_nodes} nodes. Requested '
                               f'{self.num_edges} edges, but batch has '
                               f'{num_real_edges} edges.')

        num_nodes_or_edges_positive = _all_positive(
            num_pad_nodes) or _all_positive(num_pad_edges)
        if num_pad_graphs == 0 and num_nodes_or_edges_positive:
            raise RuntimeError(
                f'Requested to pad a batch to {num_all_graphs} graphs but ' \
                f'collater got a list of {num_real_graphs} graphs and ' \
                'cannot create additional graphs to pad nodes and edges.')

        if num_pad_graphs and num_nodes_or_edges_positive:
            data = data_list[0]
            # Divide padding nodes and edges evenly between padding graphs.
            pad_nodes_by_graph, pad_edges_by_graph = _divide_evenly(
                data, num_pad_graphs, num_pad_nodes, num_pad_edges)

            data_to_pad_dict = self._create_structure_dict(data)
            for nodes, edges in zip(pad_nodes_by_graph, pad_edges_by_graph):
                padded_data = self._create_padded_data(data_list,
                                                       data_to_pad_dict, nodes,
                                                       edges)
                data_list.append(padded_data)

        batch = super().__call__(data_list)
        if self.add_masks_to_batch:
            padded_data_list = data_list[-num_pad_graphs:]
            self._add_masks(batch,
                            num_all_graphs,
                            num_real_graphs,
                            num_real_nodes=num_real_nodes,
                            num_real_edges=num_real_edges,
                            padded_data_list=padded_data_list)

        return batch

    @singledispatchmethod
    def _add_masks(self, batch, num_all_graphs, num_real_graphs, **kwargs):
        raise ValueError(f'Unsupported data type: {type(batch)}')

    @_add_masks.register(DataBatch)
    def _(self, batch: DataBatch, num_all_graphs: int, num_real_graphs: int,
          **kwargs) -> None:  # num_real_nodes: int, num_real_edges: int
        num_real_nodes = kwargs['num_real_nodes']
        num_real_edges = kwargs['num_real_edges']
        graphs_mask = torch.arange(num_all_graphs) < num_real_graphs
        nodes_mask = torch.arange(self.num_nodes) < num_real_nodes
        edges_mask = torch.arange(self.num_edges) < num_real_edges
        setattr(batch, 'graphs_mask', graphs_mask)
        setattr(batch, 'nodes_mask', nodes_mask)
        setattr(batch, 'edges_mask', edges_mask)

    @_add_masks.register(HeteroDataBatch)
    def _(self, batch: HeteroDataBatch, num_all_graphs: int,
          num_real_graphs: int,
          **kwargs) -> None:  # padded_data_list: List[HeteroDataBatch]):
        padded_data_list = kwargs['padded_data_list']
        graphs_mask = torch.arange(num_all_graphs) < num_real_graphs
        setattr(batch, 'graphs_mask', graphs_mask)

        num_padded_nodes_list = [0] * len(batch.node_stores)
        num_padded_edges_list = [0] * len(batch.edge_stores)
        for padded_data in padded_data_list:
            for idx, node_store in enumerate(padded_data.node_stores):
                num_padded_nodes_list[idx] += node_store.num_nodes
            for idx, edge_store in enumerate(padded_data.edge_stores):
                num_padded_edges_list[idx] += edge_store.num_edges

        def set_mask(stores, num_padded_list, num_attr, mask_attr):
            for attr, num_padded in zip(stores, num_padded_list):
                num_elems = getattr(attr, num_attr)
                mask = torch.arange(num_elems) < (num_elems - num_padded)
                setattr(attr, mask_attr, mask)

        set_mask(batch.node_stores, num_padded_nodes_list, 'num_nodes',
                 'nodes_mask')
        set_mask(batch.edge_stores, num_padded_edges_list, 'num_edges',
                 'edges_mask')

    def _calc_pad_limits(
            self, data_list: List[BaseData]
    ) -> Union[Tuple[int, int, int, int],
               Tuple[Dict[NodeType, int], Dict[NodeType, int],
                     Dict[NodeType, int], Dict[NodeType, int]]]:

        # Check if all elements in data_list are of the same type
        data_list_types = [type(d) for d in data_list]
        assert data_list_types[:-1] == data_list_types[1:]

        return self._calc_pad_limits_body(data_list[0], data_list)

    @singledispatchmethod
    def _calc_pad_limits_body(self, data, data_list):  # pylint: disable=unused-argument
        raise ValueError(f'Unsupported data type: {type(data)}')

    @_calc_pad_limits_body.register(Data)
    def _(self, _, data_list: List[Data]) -> Tuple[int, int, int, int]:
        def calc_pad_limits_attr(data_list, attr):
            data_num_attr = sum(getattr(d, attr) for d in data_list)
            num_pad_attr = getattr(self, attr) - data_num_attr
            return data_num_attr, num_pad_attr

        num_real_nodes, num_pad_nodes = calc_pad_limits_attr(
            data_list, 'num_nodes')
        num_real_edges, num_pad_edges = calc_pad_limits_attr(
            data_list, 'num_edges')

        return num_real_nodes, num_pad_nodes, num_real_edges, num_pad_edges

    @_calc_pad_limits_body.register(HeteroData)
    def _(self, _, data_list: List[HeteroData]
          ) -> Tuple[Dict[NodeType, int], Dict[NodeType, int],
                     Dict[EdgeType, int], Dict[EdgeType, int]]:
        real_nodes_nums = dict()
        real_edges_nums = dict()
        for data_ in data_list:
            for node_type in data_.node_types:
                real_nodes_nums[node_type] = real_nodes_nums.get(
                    node_type, 0) + data_[node_type].x.shape[0]

            for edge_type in data_.edge_types:
                real_edges_nums[edge_type] = real_edges_nums.get(
                    edge_type, 0) + data_[edge_type].edge_index.shape[1]

        pad_nodes_nums = {
            k: (self.num_nodes - v)
            for k, v in real_nodes_nums.items()
        }
        pad_edges_nums = {
            k: (self.num_edges - v)
            for k, v in real_edges_nums.items()
        }

        return real_nodes_nums, pad_nodes_nums, real_edges_nums, pad_edges_nums

    def _create_padded_data(
            self, data_list: List[BaseData],
            data_to_pad_dict: Dict[Union[NodeType, EdgeType, str], Any],
            num_nodes: int, num_edges: int) -> BaseData:
        """Create a new empty data instance (type specified based on the
        'data_list' input) padded to num_nodes and num_edges.
        """
        data = data_list[0]
        data_type = type(data)
        data_to_pad = _generate_data_to_pad.dispatch(data_type)(
            data_to_pad_dict)
        pad_op = Pad(max_num_nodes=num_nodes,
                     max_num_edges=num_edges,
                     node_pad_value=self.node_pad_value,
                     edge_pad_value=self.edge_pad_value,
                     exclude_keys=self.exclude_keys)
        padded_data = pad_op(data_to_pad)

        # Because Pad op does not pad graph values, this needs to be done
        # in a separate step.
        self._pad_graph_values(padded_data, data)

        return padded_data

    def _prune_edges(self, data_list: List[BaseData]) -> List[BaseData]:
        num_real_edges = sum(d.num_edges for d in data_list)

        # There is nothing to prune.
        if num_real_edges < self.num_edges:
            return data_list

        num_edges_to_trim = num_real_edges - self.num_edges
        edge_slices = _make_data_edge_slice_gen(data_list)

        # Prepare the mask of edges randomly chosen to remove.
        preserve_edges_mask = _create_preserve_mask(num_real_edges,
                                                    num_edges_to_trim,
                                                    edge_slices)

        # Apply the preservation masks to the data_list to finally trim edges.
        data_type = type(data_list[0])
        return _prune_data_edges.dispatch(data_type)(data_list,
                                                     preserve_edges_mask,
                                                     edge_slices)

    def _prune_nodes(self, data_list: List[BaseData]) -> List[BaseData]:
        num_real_nodes = sum(d.num_nodes for d in data_list)

        # There is nothing to prune.
        if num_real_nodes < self.num_nodes:
            return data_list

        num_graphs_to_trim = len(data_list)
        num_nodes_to_trim = num_real_nodes - self.num_nodes
        if self.num_nodes < num_graphs_to_trim:
            raise RuntimeError('Too many nodes to trim. Batch has '
                               f'{num_graphs_to_trim} graphs with '
                               f'{num_real_nodes} total nodes. Requested '
                               f'to trim it to {num_nodes_to_trim} nodes, '
                               'which would result in empty graphs.')
        node_slices = _make_node_slices_gen(data_list)

        # Prepare the mask of nodes randomly chosen to remove.
        preserve_nodes_mask = _create_preserve_mask(num_real_nodes,
                                                    num_nodes_to_trim,
                                                    node_slices)
        # Apply the preservation masks to the data_list  to finally trim nodes.
        data_type = type(data_list[0])
        return _prune_data_nodes.dispatch(data_type)(data_list,
                                                     preserve_nodes_mask,
                                                     node_slices)

    @singledispatchmethod
    def _create_structure_dict(self, data):
        """Create a dict representing the structure of the input data. Dict keys
        correspond to the 'data' keys, its values are all defaulted.
        """
        raise ValueError(f'Unsupported data type: {type(data)}')

    @_create_structure_dict.register(Data)
    def _(self, data: Data) -> Dict[NodeType, Any]:
        if self.labels_type == self.LabelsType.NODE_LVL:
            check = lambda key: (key == 'y' and self.labels_type == self.
                                 LabelsType.NODE_LVL) or (data.is_node_attr(
                                     key) or data.is_edge_attr(key))
        else:
            check = lambda key: data.is_node_attr(key) or data.is_edge_attr(key
                                                                            )

        out = dict()
        for key, val in data.to_dict().items():
            if check(key):
                out[key] = _reset_attr(val, key)
        return out

    @_create_structure_dict.register(HeteroData)
    def _(self, data: HeteroData) -> Dict[Union[NodeType, EdgeType], Any]:
        out = dict()
        for key, attr in data._global_store.to_dict().items():  # pylint: disable=protected-access
            out[key] = _reset_attr(attr)
        for key, attr in chain(data.node_items(), data.edge_items()):
            out[key] = {
                k: torch.zeros(_reset_dim(v.shape, k))
                for k, v in attr.to_dict().items()
                if isinstance(v, torch.Tensor)
            }
        return out

    @singledispatchmethod
    def _pad_graph_values(self, padded_data, original_data):
        raise ValueError(
            f'Unsupported pair of data types: {type(padded_data)}, '
            f'{type(original_data)}')

    @_pad_graph_values.register(Data)
    def _(self, padded_data: Data, original_data: Data) -> None:
        if self.labels_type == self.LabelsType.NODE_LVL:
            check = lambda key: (
                key == 'y' and self.labels_type == self.LabelsType.GRAPH_LVL
            ) or not (original_data.is_node_attr(key) or original_data.
                      is_edge_attr(key))
        else:
            check = lambda key: not (original_data.is_node_attr(key) or
                                     original_data.is_edge_attr(key))

        for key, value in original_data():
            if key in self.exclude_keys:
                continue
            if check(key):
                self._pad_graph_values_body(padded_data, original_data, key,
                                            value)

    @_pad_graph_values.register(HeteroData)
    def _(self, padded_data: HeteroData, original_data: HeteroData) -> None:
        for key, value in original_data._global_store.items():  # pylint: disable=protected-access
            if key in self.exclude_keys:
                continue
            self._pad_graph_values_body(padded_data, original_data, key, value)

    def _pad_graph_values_body(self, padded_data: BaseData,
                               original_data: BaseData, key: Any,
                               value: Any) -> None:
        if not torch.is_tensor(value):
            padded_data[key] = self.pad_graph_defaults.get(
                key, original_data[key])
        else:
            pad_shape = list(value.shape)
            pad_value = self.graph_pad_value
            padded_data[key] = value.new_full(pad_shape, pad_value)


class CombinedBatchingCollater:
    r"""Manages the combined batch size defined as :obj:`mini_batch_size *
    device_iterations * replication_factor * gradient_accumulation`.

    This class is intended to be used in combination with the
    :class:`poptorch.DataLoader`.

    Args:
        collater (Collater): The collater transforming the list of
            :class:`torch_geometric.data.Data` objects to a
            :obj:`torch_geometric.data.Batch` object.
        mini_bach_size (int, optional): The size of mini batch. If not
            provided, the length of the list provided when calling an instance
            of this class is used. (default: :obj:`None`)
    """

    def __init__(
            self,
            collater: Collater,
            mini_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.mini_batch_size = mini_batch_size
        self.collater = collater
        self.parser = types.PyGArgsParser()

    def __call__(self, batch: List[BaseData]) -> Batch:
        num_items = len(batch)
        mini_batch_size = (self.mini_batch_size
                           if self.mini_batch_size is not None else num_items)

        assert num_items % mini_batch_size == 0, \
            'Invalid batch size. ' \
            f'Got {num_items} graphs and ' \
            f'`mini_batch_size={mini_batch_size}`.'

        num_mini_batches = num_items // mini_batch_size

        def batch_slice(batch_id):
            stride = mini_batch_size
            start = batch_id * stride
            return slice(start, start + stride)

        batches = [
            self.collater(batch[batch_slice(batch_id)])
            for batch_id in range(num_mini_batches)
        ]
        batch_tensors = [
            list(self.parser.yieldTensors(batch)) for batch in batches
        ]

        return self.parser.reconstruct(
            batches[0], combine_batch_tensors_gen(batch_tensors))
