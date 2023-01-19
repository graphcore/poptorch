# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import torch
from torch_geometric.data import Batch, Data

from poppyg.pyg_collate import Collater

from . import types, utils

__all__ = ['FixedSizeCollater', 'CombinedBatchingCollater']


def make_exclude_keys(include_keys: Union[List[str], Tuple[str, ...]],
                      data: Data) -> Tuple[str, ...]:
    return tuple(set(data.keys) - set(include_keys))


def _divide_evenly(amount: int, pieces: int) -> List[int]:
    minimum = amount // pieces
    extra = amount - minimum * pieces
    return [minimum + (1 if i < extra else 0) for i in range(pieces)]


def _data_slice_gen(data_list: List[Data],
                    attr: str) -> Generator[slice, None, None]:
    start = 0
    end = 0
    for data in data_list:
        end += getattr(data, attr)
        yield slice(start, end)
        start = end


def _make_data_node_slice_gen(data_list: List[Data]
                              ) -> Callable[[], Generator[slice, None, None]]:
    return partial(_data_slice_gen, data_list, "num_nodes")


def _make_data_edge_slice_gen(data_list: List[Data]
                              ) -> Callable[[], Generator[slice, None, None]]:
    return partial(_data_slice_gen, data_list, "num_edges")


class FixedSizeCollater(Collater):
    r"""Collates a batch of graphs as a
    :class:`torch_geometric.data.Batch` of fixed-size tensors.

    Calling an instance of this class adds an additional graphs with the
    necessary number of nodes and edges to pad the batch so that tensors have
    the size corresponding to the maximum numbers of graphs, nodes and edges
    specified during initialisation.

    Calling an instance of this class can result in :obj:`RuntimeError` if
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
        self.attribute_cacher = utils.AttributeTypeCache()

    def __call__(self, data_list: List[Data]) -> Batch:

        if not isinstance(data_list, list):
            raise TypeError(f'Expected list, got {type(data_list).__name__}.')

        num_real_graphs = len(data_list)
        num_pad_graphs = 1 if self.num_graphs is None \
            else self.num_graphs - num_real_graphs
        num_all_graphs = num_real_graphs + num_pad_graphs
        num_real_nodes, num_pad_nodes, num_real_edges, num_pad_edges = \
            self._calc_pad_limits(data_list)

        if self.trim_nodes and num_pad_nodes < 0:
            data_list = self._prune_nodes(data_list)
            num_real_nodes, num_pad_nodes, num_real_edges, num_pad_edges = \
                self._calc_pad_limits(data_list)

        if self.trim_edges and num_pad_edges < 0:
            data_list = self._prune_edges(data_list)
            num_real_nodes, num_pad_nodes, num_real_edges, num_pad_edges = \
                self._calc_pad_limits(data_list)

        if num_pad_graphs < 0 or num_pad_edges < 0 or num_pad_nodes < 0:
            raise RuntimeError('Graphs in the batch are too large. Requested '
                               f'{num_all_graphs} graphs, but batch has '
                               f'{num_real_graphs} graphs. Requested '
                               f'{self.num_nodes} nodes, but batch has '
                               f'{num_real_nodes} nodes. Requested '
                               f'{self.num_edges} edges, but batch has '
                               f'{num_real_edges} edges.')

        data = data_list[0]

        if num_pad_graphs and (num_pad_nodes > 0 or num_pad_edges > 0):
            # Divide padding nodes and edges evenly between padding graphs.
            pad_nodes_by_graph = _divide_evenly(num_pad_nodes, num_pad_graphs)
            pad_edges_by_graph = _divide_evenly(num_pad_edges, num_pad_graphs)

            for nodes, edges in zip(pad_nodes_by_graph, pad_edges_by_graph):
                data_list.append(
                    data.from_dict(self._make_pad_graph(data, nodes, edges)))

        batch = super().__call__(data_list)

        if self.add_masks_to_batch:
            graphs_mask = torch.arange(num_all_graphs) < num_real_graphs
            nodes_mask = torch.arange(self.num_nodes) < num_real_nodes
            edges_mask = torch.arange(self.num_edges) < num_real_edges
            setattr(batch, 'graphs_mask', graphs_mask)
            setattr(batch, 'nodes_mask', nodes_mask)
            setattr(batch, 'edges_mask', edges_mask)

        return batch

    def _calc_pad_limits(self,
                         data_list: List[Data]) -> Tuple[int, int, int, int]:
        num_real_nodes, num_pad_nodes = self._calc_pad_limits_attr(
            data_list, "num_nodes")
        num_real_edges, num_pad_edges = self._calc_pad_limits_attr(
            data_list, "num_edges")

        return num_real_nodes, num_pad_nodes, num_real_edges, num_pad_edges

    def _calc_pad_limits_attr(self, data_list: List[Data],
                              attr: str) -> Tuple[int, int]:
        data_num_attr = sum(getattr(d, attr) for d in data_list)
        num_pad_attr = getattr(self, attr) - data_num_attr

        return data_num_attr, num_pad_attr

    def _prune_edges(self, data_list: List[Data]) -> List[Data]:
        batch_num_edges = sum(d.num_edges for d in data_list)

        # There is nothing to prune.
        if batch_num_edges < self.num_edges:
            return data_list

        num_edges_to_trim = batch_num_edges - self.num_edges

        # Randomly select edges to remove.
        prune_edges_indices = torch.randperm(
            batch_num_edges)[:num_edges_to_trim].type(torch.long)
        preserve_edges_mask = torch.ones(batch_num_edges, dtype=torch.bool)
        preserve_edges_mask[prune_edges_indices] = False

        data_slice_gen = _make_data_edge_slice_gen(data_list)()
        return [
            data.edge_subgraph(preserve_edges_mask[next(data_slice_gen)])
            for data in data_list
        ]

    def _prune_nodes(self, data_list: List[Data]) -> List[Data]:
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

        data_slice_node_gen = _make_data_node_slice_gen(data_list)
        # Prevent deletion of all nodes from a single graph.
        removable_nodes_mask = torch.ones(num_real_nodes, dtype=torch.bool)
        for data_slice in data_slice_node_gen():
            mask_slice = removable_nodes_mask[data_slice]
            mask_slice[torch.randint(high=len(mask_slice), size=(1, ))] = False
        indices = torch.arange(0, num_real_nodes)[removable_nodes_mask]

        # Randomly select nodes to remove.
        prune_nodes_indices = indices[torch.randperm(
            len(indices))][:num_nodes_to_trim].type(torch.long)
        preserve_nodes_mask = torch.ones(num_real_nodes, dtype=torch.bool)
        preserve_nodes_mask[prune_nodes_indices] = False

        data_slice_gen = data_slice_node_gen()
        return [
            data.subgraph(preserve_nodes_mask[next(data_slice_gen)])
            for data in data_list
        ]

    def _make_pad_graph(self, data: Data, nodes: int, edges: int) -> Data:
        new_data = {}
        for key, value in data():
            if key not in self.exclude_keys and not torch.is_tensor(value):
                if key == 'num_nodes':
                    new_data[key] = nodes
                else:
                    new_data[key] = self.pad_graph_defaults.get(key, value)
                continue

            dim = data.__cat_dim__(key, value)
            pad_shape = list(value.shape)

            if self.attribute_cacher.is_node_attr(data, key):
                pad_shape[dim] = nodes
                pad_value = self.node_pad_value
            elif self.attribute_cacher.is_edge_attr(data, key):
                pad_shape[dim] = edges

                if key == 'edge_index':
                    # Padding edges are self-loops on the first padding
                    # node.
                    pad_value = 0
                else:
                    pad_value = self.edge_pad_value
            else:
                pad_value = self.graph_pad_value
            new_data[key] = value.new_full(pad_shape, pad_value)

        return new_data


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

    def __call__(self, batch: List[Data]) -> Batch:
        num_items = len(batch)
        mini_batch_size = (self.mini_batch_size
                           if self.mini_batch_size is not None else num_items)

        assert num_items % mini_batch_size == 0, \
            'Invalid batch size. ' \
            f'Got {num_items} graphs and' \
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

        combined_batch_tensors = [
            torch.stack([
                batch_tensors[batch_id][tensor_id]
                for batch_id in range(len(batches))
            ]) for tensor_id in range(len(batch_tensors[0]))
        ]

        return self.parser.reconstruct(batches[0], combined_batch_tensors)
