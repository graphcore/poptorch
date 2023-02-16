# Copyright (c) 2022-2023 Graphcore Ltd. All rights reserved.

import numbers
from functools import lru_cache
from typing import Any, Generator, Iterable, Iterator, List, Optional, Union

from torch.utils.data.sampler import RandomSampler, Sampler
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData

from poptorch_geometric.collate import FixedSizeCollater

__all__ = ['FixedBatchSampler', 'make_fixed_batch_generator']


class FixedBatchSampler(Sampler[List[int]]):
    r""""Wraps another sampler to yield a mini-batch of indices.
    :class:`FixedBatchSampler` forms batches by adding graphs to the batch one
    at a time without exceeding the maximum number of nodes, edges, or graphs.
    This gives similar results to packing without requiring the dataset to be
    preprocessed.

    Args:
        data_source (torch_geometric.data.Dataset): The source of data to
            process.
        num_graphs (int): Number of graphs in a batch.
        num_nodes (int, optional): Number of nodes in a batch.
            (default: :obj:`None`)
        num_edges (int, optional): Number of edges in a batch.
            (default: :obj:`None`)
        sampler (Sampler or Iterable, optional): Base sampler. Can be any
            iterable object. (default: RandomSampler(data_source))
        allow_skip_data (bool, optional): Allow skip :obj:`data_source` item,
            otherwise throw :class:`RuntimeError` when the sampler is not able
            to form a single item batch from :obj:`data_source`, because
            the iterated data exceeds the maximum batch requirements.
            (default :obj:`False`)
    """

    def __init__(self,
                 data_source: Dataset,
                 num_graphs: int,
                 num_nodes: Optional[int] = None,
                 num_edges: Optional[int] = None,
                 sampler: Optional[Union[Sampler[int], Iterable[int]]] = None,
                 allow_skip_data: Optional[bool] = False) -> None:
        super().__init__(data_source)
        self._validate(sampler, num_nodes, num_edges, num_graphs)

        self.data_source = data_source
        self.num_graphs = num_graphs

        self.num_nodes = num_nodes
        if num_nodes is None:
            self.num_nodes = max(data.num_nodes
                                 for data in data_source) * num_graphs

        self.num_edges = num_edges
        if num_edges is None:
            self.num_edges = max(data.num_edges
                                 for data in data_source) * num_graphs

        self.sampler = sampler if sampler is not None else \
            RandomSampler(data_source)
        self.allow_skip_data = allow_skip_data

    def _validate(self, sampler, num_nodes, num_edges, num_graphs):
        if sampler is not None and len(sampler) == 0:
            raise ValueError(
                'Invalid `sampler` parameter, `len(sampler) == 0`.')

        def validate_batch_limit(param, param_name, limit=1):
            if param is not None and param < limit:
                raise ValueError(f'Invalid `{param_name}` parameter, '
                                 f'{param_name} should be at least {limit}.')

        if num_graphs is None:
            raise ValueError(
                'Invalid `num_graphs` parameter, `num_graphs` is None.')

        validate_batch_limit(num_graphs, 'num_graphs', 1)
        validate_batch_limit(num_nodes, 'num_nodes', num_graphs)
        validate_batch_limit(num_edges, 'num_edges', num_graphs)

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
        for idx in self.sampler:
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
                    raise RuntimeError(f'Dataset[{idx}] {data} is not '
                                       'appendable to empty batch with  '
                                       'following configuration: { number of '
                                       f'graphs: {self.num_graphs}, number of '
                                       f'nodes: {self.num_nodes}, number of '
                                       f'edges: {self.num_edges} }}.')
                break

        if not batch.empty():
            yield batch.indices

    def _has_space(self, batch: _Batch, data: BaseData) -> bool:
        next_nodes = data.num_nodes
        next_edges = data.num_edges

        nodes_left = self.num_nodes - (batch.num_nodes + next_nodes)
        edges_left = self.num_edges - (batch.num_edges + next_edges)
        graphs_left = self.num_graphs - (batch.num_graphs + 1)

        graph_fits = nodes_left >= 0 and edges_left >= 0 and \
            graphs_left >= 0
        has_space_for_padding = nodes_left >= graphs_left and \
            edges_left >= graphs_left

        has_space = graph_fits and has_space_for_padding
        return has_space

    @lru_cache(maxsize=128)
    def __len__(self) -> int:
        if isinstance(self.sampler, RandomSampler):
            raise NotImplementedError(
                '`self.sampler` is `RandomSampler`. '
                f'{self.__class__.__name__} length is non deterministic.')

        return len(list(self.__iter__()))


def make_fixed_batch_generator(batch_sampler: Sampler[List[int]],
                               data_source: Optional[Dataset] = None,
                               num_graphs: Optional[int] = None,
                               num_nodes: Optional[int] = None,
                               num_edges: Optional[int] = None
                               ) -> Generator[Batch, None, None]:
    r"""Creates a generator that transforms the list of the :obj:`data_source`
    indices into :class:`torch_geometric.data.Batch` object.

    Args:
        batch_sampler (Sampler): The sampler that defines the strategy to draw
            a batch of indices from the :obj:`data_source`.
        data_source (Dataset, optional): The source of data to process.
        num_graphs (int, optional): Number of graphs in a batch.
        num_nodes (int, optional): Number of nodes in a batch.
        num_edges (int, optional): Number of edges in a batch.

    Returns:
        Created generator.
    """

    def get_param(param: Any, param_name: str) -> Any:
        nonlocal batch_sampler
        sampler_attr = getattr(batch_sampler, param_name, None)

        if param is None and sampler_attr is None:
            raise ValueError(f'Passed parameter `{param_name}` is None '
                             f'and `batch_sampler` does not have an '
                             f'attribute called `{param_name}`.')

        is_number = isinstance(sampler_attr, numbers.Number)
        if param is None:
            # Add 1 to leave space for padding graph, node and edge.
            return sampler_attr + 1 if is_number else sampler_attr

        if sampler_attr is None:
            return param

        if is_number and sampler_attr + 1 > param:
            raise ValueError(f'When provided, parameter `{param_name}` (= '
                             f'{param}) should be greater than sampler\'s ' \
                             f'`{param_name}` attribute (= {sampler_attr}) ' \
                             f'to account for padding and it is not.')

        return param

    # If using values from the sampler, add at least one padding graph, node
    # and edge.
    num_graphs = get_param(num_graphs, 'num_graphs')
    num_nodes = get_param(num_nodes, 'num_nodes')
    num_edges = get_param(num_edges, 'num_edges')
    data_source = get_param(data_source, 'data_source')

    collater = FixedSizeCollater(num_nodes, num_edges, num_graphs)

    for batch_sample in batch_sampler:
        yield collater([data_source[idx] for idx in batch_sample])
