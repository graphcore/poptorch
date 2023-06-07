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

from poptorch_geometric.fixed_size_options import FixedSizeOptions
from poptorch_geometric.pyg_collate import Collater
from poptorch_geometric.common import DataBatch, HeteroDataBatch

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


def data_slice_gen(num_list: List[int]) -> Generator[slice, None, None]:
    start = 0
    end = 0
    for num in num_list:
        end += num
        yield slice(start, end)
        start = end


def create_slices_and_preserve_mask(
        max_num: int, num_list: List[int]
) -> Tuple[Generator[slice, None, None], List[bool]]:
    num_real = sum(num_list)

    # There is nothing to prune.
    if num_real < max_num:
        return None, None

    num_to_trim = num_real - max_num

    slices = list(data_slice_gen(num_list))

    # Prepare the mask of randomly chosen to remove.
    preserve_mask = _create_preserve_mask(num_real, num_to_trim, slices)

    return slices, preserve_mask


@singledispatch
def _any_negative(value: int) -> bool:
    return value < 0


@_any_negative.register(dict)
def _(value: dict) -> bool:
    return any(v < 0 for v in value.values())


@singledispatch
def _any_positive(value: int) -> bool:
    return value > 0


@_any_positive.register(dict)
def _(value: dict) -> bool:
    return any(v > 0 for v in value.values())


@singledispatch
def _check_if_over_size(num_pad: int, num_total: int, type_str: str,
                        oversize_error: str):
    if _any_negative(num_pad):
        raise RuntimeError(
            oversize_error.format(type_str=type_str,
                                  trim_fn=f"trim_{type_str}",
                                  type_value=num_total))


@_check_if_over_size.register(dict)
def _(num_pad: dict, num_total: dict, type_str: str, oversize_error: str):
    for k, v in num_pad.items():
        if v < 0:
            raise RuntimeError(
                oversize_error.format(type_str=f"{k} {type_str}",
                                      trim_fn=f"trim_{type_str}",
                                      type_value=num_total[k]))


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
        fixed_size_options (FixedSizeOptions, optional): A
            :py:class:`poptorch_geometric.fixed_size_options.FixedSizeOptions`
            object which holds the maximum number of nodes, edges and other
            options required to pad the batches, produced by collater,
            to a fixed size.
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
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (list or tuple, optional): The keys to exclude
            from the graphs in the output batch. (default: :obj:`None`)
    """

    def __init__(
            self,
            fixed_size_options: FixedSizeOptions,
            add_masks_to_batch: Optional[bool] = False,
            trim_nodes: Optional[bool] = False,
            trim_edges: Optional[bool] = False,
            follow_batch: Optional[Union[List[str], Tuple[str, ...]]] = None,
            exclude_keys: Optional[Union[List[str], Tuple[str, ...]]] = None,
    ) -> None:
        super().__init__(follow_batch, exclude_keys)
        self.opts = fixed_size_options
        self.add_masks_to_batch = add_masks_to_batch
        self.trim_nodes = trim_nodes
        self.trim_edges = trim_edges
        self.labels_type = None

    class LabelsType(Enum):
        GRAPH_LVL = 0
        NODE_LVL = 1

    def __call__(self, data_list: List[BaseData]) -> Batch:
        if not self.opts.is_hetero() and isinstance(data_list[0], HeteroData):
            self.opts.to_hetero(data_list[0].node_types,
                                data_list[0].edge_types)

        if not isinstance(data_list, list):
            raise TypeError(f'Expected list, got {type(data_list).__name__}.')

        if (isinstance(data_list[0], Data) and hasattr(data_list[0], 'y')
                and data_list[0].y is not None):
            y0_equal_num_nodes = all(data.y.shape[0] == data.num_nodes
                                     for data in data_list)
            y0_equal_ones = all(data.y.shape[0] == 1 for data in data_list)

            if y0_equal_num_nodes and not y0_equal_ones:
                self.labels_type = self.LabelsType.NODE_LVL
            elif y0_equal_ones and not y0_equal_num_nodes:
                self.labels_type = self.LabelsType.GRAPH_LVL
            else:
                assert False, "Incorrect input data. The size of the shape" \
                              "of labels `y` must be either the number" \
                              "of nodes or the number of graphs"

        num_real_graphs = len(data_list)
        num_pad_graphs = self.opts.num_graphs - num_real_graphs

        if num_pad_graphs < 0:
            raise RuntimeError(
                "The maximum number of graphs requested doesn't allocate"
                " enough room for all the graphs in the batch plus at least"
                " one extra graph required for padding the batch to a fixed"
                " size. The number of graphs received for batching is"
                f" {num_real_graphs + 1}, including at least one padding"
                " graph, but space for only"
                f" {num_pad_graphs + num_real_graphs} graphs has been"
                " requested.")
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

        oversize_error = (
            "The fixed sizes given don't allocate enough space for the"
            " number of {type_str} required to fit"
            f" {num_real_graphs} sample(s) into a batch"
            f" ({num_pad_graphs + num_real_graphs} including extra padded"
            " graph(s)). Increase the maximum number of {type_str}, currently"
            " set to {type_value}, or set `{trim_fn}` to remove any"
            " excess {type_str} to achieve the given maximum number of"
            " {type_str}.")

        _check_if_over_size(num_pad_nodes, self.opts.num_nodes, "nodes",
                            oversize_error)
        _check_if_over_size(num_pad_edges, self.opts.num_edges, "edges",
                            oversize_error)

        num_nodes_or_edges_positive = _any_positive(
            num_pad_nodes) or _any_positive(num_pad_edges)
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
        nodes_mask = torch.arange(self.opts.num_nodes) < num_real_nodes
        edges_mask = torch.arange(self.opts.num_edges) < num_real_edges
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
        def calc_pad_limits_attr(data_list, attr, fixed_size):
            data_num_attr = sum(getattr(d, attr) for d in data_list)
            num_pad_attr = fixed_size - data_num_attr
            return data_num_attr, num_pad_attr

        num_real_nodes, num_pad_nodes = calc_pad_limits_attr(
            data_list, 'num_nodes', self.opts.num_nodes)
        num_real_edges, num_pad_edges = calc_pad_limits_attr(
            data_list, 'num_edges', self.opts.num_edges)

        return num_real_nodes, num_pad_nodes, num_real_edges, num_pad_edges

    @_calc_pad_limits_body.register(HeteroData)
    def _(self, _, data_list: List[HeteroData]
          ) -> Tuple[Dict[NodeType, int], Dict[NodeType, int],
                     Dict[EdgeType, int], Dict[EdgeType, int]]:
        real_nodes_nums = dict()
        pad_nodes_nums = dict()
        real_edges_nums = dict()
        pad_edges_nums = dict()
        for data_ in data_list:
            for node_type in data_.node_types:
                num_real_nodes = real_nodes_nums.get(
                    node_type, 0) + data_[node_type].x.shape[0]
                real_nodes_nums[node_type] = num_real_nodes

                if isinstance(self.opts.num_nodes, dict):
                    assert node_type in self.opts.num_nodes, (
                        f"Node type {node_type} exists in the data"
                        " but not in the fixed size options. Ensure"
                        " your fixed size options specify a `num_nodes`"
                        f" for node type {node_type}.")
                    num_pad_nodes = self.opts.num_nodes[
                        node_type] - num_real_nodes
                else:
                    num_pad_nodes = self.opts.num_nodes - num_real_nodes
                pad_nodes_nums[node_type] = num_pad_nodes

            for edge_type in data_.edge_types:
                num_real_edges = real_edges_nums.get(
                    edge_type, 0) + data_[edge_type].edge_index.shape[1]
                real_edges_nums[edge_type] = num_real_edges

                if isinstance(self.opts.num_edges, dict):
                    assert edge_type in self.opts.num_edges, (
                        f"Edge type {edge_type} exists in the data"
                        " but not in the fixed size options. Ensure"
                        " your fixed size options specify a `num_edges`"
                        f" for edge type {edge_type}.")
                    num_pad_edges = self.opts.num_edges[
                        edge_type] - num_real_edges
                else:
                    num_pad_edges = self.opts.num_edges - num_real_edges
                pad_edges_nums[edge_type] = num_pad_edges

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
                     node_pad_value=self.opts.node_pad_value,
                     edge_pad_value=self.opts.edge_pad_value,
                     exclude_keys=self.exclude_keys)
        padded_data = pad_op(data_to_pad)

        # Because Pad op does not pad graph values, this needs to be done
        # in a separate step.
        self._pad_graph_values(padded_data, data)

        return padded_data

    def _prune_edges(self, data_list):
        return self._prune_edges_body(data_list[0], data_list)

    @singledispatchmethod
    def _prune_edges_body(self, data, data_list):  # pylint: disable=unused-argument
        raise ValueError(f'Unsupported data type: {type(data)}')

    @_prune_edges_body.register(Data)
    def _(self, _, data_list: List[Data]) -> List[Data]:
        edge_slices, preserve_edges_mask = create_slices_and_preserve_mask(
            self.opts.num_edges, [d.num_edges for d in data_list])

        # There is nothing to prune.
        if edge_slices is None:
            return data_list

        # Apply the preservation masks to the data_list to finally trim edges.
        return [
            data.edge_subgraph(preserve_edges_mask[slc])
            for data, slc in zip(data_list, edge_slices)
        ]

    @_prune_edges_body.register(HeteroData)
    def _(self, data: HeteroData,
          data_list: List[HeteroData]) -> List[HeteroData]:
        edge_types = data.edge_types
        preserve_edges_masks_dict = dict()
        edge_slices_dict = dict()

        for edge_type in edge_types:
            edge_slices, preserve_edges_mask = create_slices_and_preserve_mask(
                self.opts.num_edges[edge_type],
                [d[edge_type].edge_index.shape[1] for d in data_list])
            preserve_edges_masks_dict[edge_type] = preserve_edges_mask
            edge_slices_dict[edge_type] = edge_slices

        return [
            data.edge_subgraph({
                edge_type: preserve_edges_masks_dict[edge_type][
                    edge_slices_dict[edge_type][idx]]
                for edge_type in edge_types
                if edge_slices_dict[edge_type] is not None
            }) for idx, data in enumerate(data_list)
        ]

    def _prune_nodes(self, data_list):
        return self._prune_nodes_body(data_list[0], data_list)

    @singledispatchmethod
    def _prune_nodes_body(self, data, data_list):  # pylint: disable=unused-argument
        raise ValueError(f'Unsupported data type: {type(data)}')

    @_prune_nodes_body.register(Data)
    def _(self, _, data_list: List[BaseData]) -> List[BaseData]:
        num_graphs_to_trim = len(data_list)
        if self.opts.num_nodes < num_graphs_to_trim:
            raise RuntimeError(
                f'The number of nodes to trim to ({self.opts.num_nodes})'
                ' is less than the number of graphs in the batch'
                f' ({num_graphs_to_trim}), which would result in empty'
                ' graphs.')

        nodes_slices, preserve_nodes_mask = create_slices_and_preserve_mask(
            self.opts.num_nodes, [d.num_nodes for d in data_list])

        # There is nothing to prune.
        if nodes_slices is None:
            return data_list

        # Apply the preservation masks to the data_list  to finally trim nodes.
        return [
            data.subgraph(preserve_nodes_mask[slice])
            for data, slice in zip(data_list, nodes_slices)
        ]

    @_prune_nodes_body.register(HeteroData)
    def _(self, data: HeteroData,
          data_list: List[HeteroData]) -> List[HeteroData]:
        node_types = data.node_types
        num_graphs_to_trim = len(data_list)
        preserve_nodes_masks_dict = dict()
        node_slices_dict = dict()

        for node_type in node_types:
            if self.opts.num_nodes[node_type] < num_graphs_to_trim:
                raise RuntimeError(
                    f'The number of nodes to trim to ({self.opts.num_nodes})'
                    f' for node type {node_type} is less than the number'
                    f' of graphs in the batch ({num_graphs_to_trim}), which'
                    ' would result in empty graphs.')
            node_slices, preserve_nodes_mask = create_slices_and_preserve_mask(
                self.opts.num_nodes[node_type],
                [d[node_type].num_nodes for d in data_list])
            preserve_nodes_masks_dict[node_type] = preserve_nodes_mask
            node_slices_dict[node_type] = node_slices

        return [
            data.subgraph({
                node_type: preserve_nodes_masks_dict[node_type][
                    node_slices_dict[node_type][idx]]
                for node_type in data.node_types
                if node_slices_dict[node_type] is not None
            }) for idx, data in enumerate(data_list)
        ]

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
                k: torch.zeros(_reset_dim(v.shape, k),
                               dtype=data[key][k].dtype)
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
            padded_data[key] = self.opts.pad_graph_defaults.get(
                key, original_data[key])
        else:
            pad_shape = list(value.shape)
            pad_value = self.opts.graph_pad_value
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
