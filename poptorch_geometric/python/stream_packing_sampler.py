# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from functools import lru_cache
from typing import Iterable, Iterator, List, Optional, Union

from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData

__all__ = ['StreamPackingSampler']


class StreamPackingSampler(Sampler[List[int]]):
    r"""Wraps a sampler to generate a mini-batch of graphs with potentially
    varying batch sizes.
    :py:class:`StreamPackingSampler` creates batches by adding one graph at
    a time to the batch one at a time without exceeding the maximum number
    of nodes, edges, or graphs. This gives similar results to packing without
    requiring the dataset to be preprocessed.

    Args:
        data_source (torch_geometric.data.Dataset): The data source to
            process.
        max_num_graphs (int): The maximum number of graphs to include in a
            batch.
        max_num_nodes (int, optional): The maximum number of nodes allowed in a
            batch. (default: :obj:`None`)
        max_num_edges (int, optional): The maximum number of edges allowed in a
            batch. (default: :obj:`None`)
        base_sampler (Sampler or Iterable, optional): The base sampler used to
            sample the graphs before packing them into a batch. This can be any
            iterable object. (default: SequentialSampler(data_source))
        allow_skip_data (bool, optional): If true, allows for a skip
            :obj:`data_source` item to be skipped. Otherwise, a
            :py:exc:`RuntimeError` will be thrown when the sampler is not able
            to form a single item batch from :obj:`data_source`, because
            the iterated data exceeds the maximum batch requirements.
            (default :obj:`False`)
    """

    def __init__(
            self,
            data_source: Dataset,
            max_num_graphs: int,
            max_num_nodes: Optional[int] = None,
            max_num_edges: Optional[int] = None,
            base_sampler: Optional[Union[Sampler[int], Iterable[int]]] = None,
            allow_skip_data: Optional[bool] = False) -> None:
        super().__init__(data_source)
        self._validate(base_sampler, max_num_nodes, max_num_edges,
                       max_num_graphs)

        self.data_source = data_source
        self.max_num_graphs = max_num_graphs

        self.max_num_nodes = max_num_nodes
        if max_num_nodes is None:
            self.max_num_nodes = max(data.num_nodes
                                     for data in data_source) * max_num_graphs

        self.max_num_edges = max_num_edges
        if max_num_edges is None:
            self.max_num_edges = max(data.num_edges
                                     for data in data_source) * max_num_graphs

        self.base_sampler = base_sampler if base_sampler is not None else \
            SequentialSampler(data_source)
        self.allow_skip_data = allow_skip_data

    def _validate(self, sampler, max_num_nodes, max_num_edges, max_num_graphs):
        if sampler is not None and len(sampler) == 0:
            raise ValueError(
                f'The `sampler` {sampler} provided is invalid,'
                ' the length of the sampler must be greater than 0.')

        def validate_batch_limit(param, param_name, limit=1):
            if param is not None and param < limit:
                raise ValueError(
                    f'Invalid value for `{param_name}` parameter, '
                    f'{param_name} should be at least greater '
                    f' than {limit}.')

        if max_num_graphs is None:
            raise ValueError('Invalid value for `max_num_graphs` parameter.'
                             ' `max_num_graphs` must be an integer of at least'
                             ' 1, it is None.')

        validate_batch_limit(max_num_graphs, 'max_num_graphs', 1)
        validate_batch_limit(max_num_nodes, 'max_num_nodes', max_num_graphs)
        validate_batch_limit(max_num_edges, 'max_num_edges', max_num_graphs)

    class _Batch:
        def __init__(self) -> None:
            self.indices: List[int] = []
            self.num_nodes = 0
            self.num_edges = 0
            self.num_graphs = 0

        def append(self, idx: int, data: BaseData) -> None:
            self.indices.append(idx)
            self.num_nodes += data.num_nodes
            self.num_edges += data.num_edges
            self.num_graphs += 1

        def empty(self) -> bool:
            return len(self.indices) == 0

        def __repr__(self) -> str:
            return f'Batch{{ indices: {self.indices}, ' \
                   f'num_nodes: {self.num_nodes}, ' \
                   f'num_edges: {self.num_edges}, ' \
                   f'num_graphs: {self.num_graphs} }}'

    def __iter__(self) -> Iterator[List[int]]:
        batch = self._Batch()
        for idx in self.base_sampler:
            data = self.data_source[idx]
            is_data_appendable = True

            while True:
                if self._has_space(batch, data):
                    batch.append(idx, data)
                elif not batch.empty():
                    yield batch.indices
                    batch = self._Batch()
                    continue
                else:
                    is_data_appendable = False

                if not self.allow_skip_data and not is_data_appendable:
                    raise RuntimeError(
                        'The maximum number of graphs, nodes or edges'
                        ' specified is too small to fit in the single sample'
                        f' {idx} with {data.num_nodes} nodes and'
                        f' {data.num_edges} edges. The maximum number of graphs'
                        f' specified is {self.max_num_graphs}, the maximum'
                        f' number of nodes is {self.max_num_nodes} and the'
                        f' maximum number of edges is {self.max_num_edges}.'
                        ' If this is intended, use `allow_skip_data` to'
                        ' enable this sample to be completely skipped'
                        f' from batching. The sample is {data}.')
                break

        if not batch.empty():
            yield batch.indices

    def _has_space(self, batch: _Batch, data: BaseData) -> bool:
        next_nodes = data.num_nodes
        next_edges = data.num_edges

        nodes_left = self.max_num_nodes - (batch.num_nodes + next_nodes)
        edges_left = self.max_num_edges - (batch.num_edges + next_edges)
        graphs_left = self.max_num_graphs - (batch.num_graphs + 1)

        graph_fits = nodes_left >= 0 and edges_left >= 0 and \
            graphs_left >= 0
        has_space_for_padding = nodes_left >= graphs_left and \
            edges_left >= graphs_left

        has_space = graph_fits and has_space_for_padding
        return has_space

    @lru_cache(maxsize=128)
    def __len__(self) -> int:
        if isinstance(self.base_sampler, RandomSampler):
            raise NotImplementedError(
                f'{self.__class__.__name__} length (`__len__`) cannot'
                ' be determined. The base sampler used is an instance of'
                '`RandomSampler`, which will result in'
                f' {self.__class__.__name__} producing a nondeterministic'
                ' number of batches. When using this sampler with stream'
                ' packing avoid requiring the length.')

        return len(list(self.__iter__()))
