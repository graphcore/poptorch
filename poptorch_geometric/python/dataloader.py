# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from __future__ import annotations  # noqa: F407

from typing import Dict, Iterable, List, Optional, Tuple, Union

from torch.utils.data.sampler import Sampler
from torch_geometric.data import Dataset

import poptorch
from poptorch_geometric.collate import CombinedBatchingCollater
from poptorch_geometric.pyg_dataloader import CustomFixedSizeDataLoader as PyGCustomFixedSizeDataLoader
from poptorch_geometric.pyg_dataloader import DataLoader as PyGDataLoader
from poptorch_geometric.pyg_dataloader import FixedSizeDataLoader as PyGFixedSizeDataLoader


class DataLoader(PyGDataLoader, poptorch.DataLoader):
    r"""A data loader which merges data objects from a
    :py:class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :py:class:`~torch_geometric.data.Data` or
    :py:class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        options (poptorch.Options, optional): The additional PopTorch options
            to be passed to :py:class:`poptorch.DataLoader`.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :py:class:`poptorch.DataLoader`.
    """

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int = 1,
            shuffle: bool = False,
            follow_batch: Optional[Union[List[str], Tuple[str, ...]]] = None,
            exclude_keys: Optional[Union[List[str], Tuple[str, ...]]] = None,
            options: Optional[poptorch.Options] = None,
            **kwargs,
    ):
        self.batch_size = batch_size

        if options is None:
            options = poptorch.Options()

        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         follow_batch=follow_batch,
                         exclude_keys=exclude_keys,
                         options=options,
                         **kwargs)

    def _create_collater(self, **collater_args):
        base_collater = super()._create_collater(**collater_args)
        return CombinedBatchingCollater(mini_batch_size=self.batch_size,
                                        collater=base_collater)


class CustomFixedSizeDataLoader(PyGCustomFixedSizeDataLoader,
                                poptorch.DataLoader):
    r"""A data loader which merges data objects from a
    :py:class:`poptorch.Dataset` to a mini-batch and pads node and edge features
    so tensors across all batches have constant shapes.
    Data objects can be either of type :py:class:`~torch_geometric.data.Data` or
    :py:class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        num_nodes (int, optional): The total number of nodes in the padded
            batch. If the value is not provided, it will be set to the
            maximum number of nodes times the batch size. For maximum
            performance it is recommended to tune that value per specific
            use case. (default: `None`)
        batch_size (int, optional): The number of samples per batch to load.
            This should be at least :obj:`2` to allow for creating at least
            one padding graph. (default: :obj:`None`)
        batch_sampler (Sampler, optional): Batch sampler to yield a mini-batch
            of indices. If :obj:`batch_sampler` is specified, the
            :obj:`batch_size` and :obj:`shuffle` arguments do not have any
            effect and are omited. (default: :obj:`None`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (list or tuple, optional): The keys to exclude from the
            input data object. (default: :obj:`None`)
        collater_args (dict, optional): The additional arguments passed to
            :py:class:`poptorch_geometric.collate.FixedSizeCollater`. They
            should not contain :obj:`num_nodes`, :obj:`follow_batch` and
            :obj:`exclude_keys` as those should be passed directly to the
            initializer method. (default: :obj:`None`)
        options (poptorch.Options, optional): The additional PopTorch options
            to be passed to :py:class:`poptorch.DataLoader`.
            (default: :obj:`None`)
        **kwargs (optional): The additional arguments of
            :py:class:`poptorch.DataLoader`.
    """

    def __init__(
            self,
            dataset: Dataset,
            num_nodes: Optional[int] = None,
            batch_size: Optional[int] = None,
            batch_sampler: Optional[Sampler[List[int]]] = None,
            shuffle: bool = False,
            follow_batch: Optional[Union[List[str], Tuple[str, ...]]] = None,
            exclude_keys: Optional[Union[List[str], Tuple[str, ...]]] = None,
            collater_args: Optional[Dict[str, Union[int, float]]] = None,
            options: Optional[poptorch.Options] = None,
            **kwargs,
    ):
        self.batch_sampler = batch_sampler
        if options is None:
            # Create IPU default options
            options = poptorch.Options()
        super().__init__(dataset=dataset,
                         num_nodes=num_nodes,
                         batch_size=batch_size,
                         batch_sampler=batch_sampler,
                         shuffle=shuffle,
                         follow_batch=follow_batch,
                         exclude_keys=exclude_keys,
                         collater_args=collater_args,
                         options=options,
                         **kwargs)

    def _create_collater(self, **collater_args):
        base_collater = super()._create_collater(**collater_args)
        if self.batch_sampler is not None:
            mini_batch_size = None
        else:
            mini_batch_size = self.padded_batch_size - 1
        return CombinedBatchingCollater(mini_batch_size=mini_batch_size,
                                        collater=base_collater)


class FixedSizeDataLoader(PyGFixedSizeDataLoader, CustomFixedSizeDataLoader):
    r"""A data loader which merges data objects from
    :py:class:`poptorch.Dataset` to a mini-batch and pads node and
    edge features so tensors across all batches have the same shapes.
    The data loader uses
    :py:class:`poptorch_geometric.stream_packing_sampler.StreamPackingSampler`
    to select the samples that will be batched together.

    If not specified, :obj:`num_nodes` and :obj:`num_edges` are set to the
    batch size times the maximum number of nodes and edges, respectively.

    Args:
        dataset (Dataset): The :py:class:`~torch_geometric.data.Dataset`
            instance from which to load the graph examples for the IPU.
        num_nodes (int, optional): Number of nodes in a batch.
            (default: :obj:`None`)
        num_edges (int, optional): Number of edges in a batch.
            (default: :obj:`None`)
        batch_size (int, optional): How many graph examples to load in each
            batch. This should be at least :obj:`2` to allow for creating at
            least one padding graph. (default: :obj:`2`)
        options (poptorch.Options, optional): The :py:class:`poptorch.Options`
            used by the :py:class:`poptorch.DataLoader`. Will use the default
            options if not provided. (default: :obj:`None`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (list or tuple, optional): Keys to exclude from the
            batch. (default: :obj:`None`)
        collater_args (dict, optional): The additional arguments passed to
            :py:class:`poptorch_geometric.collate.FixedSizeCollater`. They
            should not contain :obj:`num_nodes`, :obj:`follow_batch` and
            :obj:`exclude_keys` as those should be passed directly to the
            initializer method. (default: :obj:`None`)
        sampler (Sampler or Iterable, optional): Base sampler. Can be any
            iterable object. (default: :obj:`None`)
        allow_skip_data (bool, optional): Allow skip :obj:`data_source` item,
            otherwise throw :py:exc:`RuntimeError` when the sampler is not able
            to form a single item batch from :obj:`data_source`, because
            :obj:`Data` exceeds the maximum batch requirements.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :py:class:`poptorch.DataLoader`.
    """

    def __init__(
            self,
            dataset: Dataset,
            num_nodes: Optional[int] = None,
            num_edges: Optional[int] = None,
            batch_size: int = 2,
            options: Optional[poptorch.Options] = None,
            follow_batch: Optional[Union[List[str], Tuple[str, ...]]] = None,
            exclude_keys: Optional[Union[List[str], Tuple[str, ...]]] = None,
            collater_args: Optional[Dict[str, Union[int, float]]] = None,
            sampler: Optional[Union[Sampler[int], Iterable[int]]] = None,
            allow_skip_data: bool = False,
            **kwargs,
    ) -> None:

        super().__init__(dataset=dataset,
                         num_nodes=num_nodes,
                         num_edges=num_edges,
                         batch_size=batch_size,
                         follow_batch=follow_batch,
                         exclude_keys=exclude_keys,
                         collater_args=collater_args,
                         sampler=sampler,
                         allow_skip_data=allow_skip_data,
                         options=options,
                         **kwargs)
