# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from __future__ import annotations  # noqa: F407

from typing import Dict, Iterable, List, Optional, Tuple, Union

from torch.utils.data.sampler import Sampler
from torch_geometric.data import Dataset

import poptorch
from poppyg.collate import CombinedBatchingCollater
from poppyg.pyg_dataloader import FixedSizeDataLoader as PyGFixedSizeDataLoader
from poppyg.pyg_dataloader import TorchDataLoaderMeta
from poppyg.pyg_dataloader import \
    create_fixed_batch_dataloader as pyg_create_fixed_batch_dataloader


class PopTorchDataLoaderMeta(TorchDataLoaderMeta):
    r"""Injects the :obj:`poptorch.DataLoader` class as a base class of class
    that uses the `PopTorchDataLoaderMeta` metaclass.
    """
    base_loader = poptorch.DataLoader


# pylint: disable=invalid-metaclass
class FixedSizeDataLoader(PyGFixedSizeDataLoader,
                          metaclass=PopTorchDataLoaderMeta):
    r"""A data loader which merges data objects from a
    :class:`poptorch.Dataset` to a mini-batch and pads node and edge features
    so tensors across all batches have constant shapes.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        num_nodes (int): The total number of nodes in the padded batch.
        batch_size (int, optional): The number of samples per batch to load.
            (default: :obj:`1`)
        batch_sampler (Sampler, optional): Batch sampler to yield a mini-batch
            of indices. If :obj:`batch_sampler` is specified, the
            :obj:`batch_size` and :obj:`shuffle` arguments do not have any
            effect and are omited. (default: :obj:`None`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        exclude_keys (list or tuple, optional): The keys to exclude from the
            input data object. (default: :obj:`None`)
        collater_args (dict, optional): The additional arguments passed to
            :class:`FixedSizeCollater`. They should not contain
            :obj:`num_nodes` or :obj:`exclude_keys` as those should be passed
            directly to the initializer method. (default :obj:`None`)
        options (poptorch.Options, optional): The additional PopTorch options
            to be passed to :obj:`poptorch.DataLoader`. (default: :obj:`None`)
        **kwargs (optional): The additional arguments of
            :class:`poptorch.DataLoader`.
    """

    def __init__(
            self,
            dataset: Dataset,
            num_nodes: int,
            batch_size: int = 1,
            batch_sampler: Optional[Sampler[List[int]]] = None,
            shuffle: bool = False,
            exclude_keys: Optional[Union[List[str], Tuple[str, ...]]] = None,
            collater_args: Optional[Dict[str, Union[int, float]]] = None,
            options: Optional[poptorch.Options] = None,
            **kwargs,
    ):
        self.batch_size = batch_size
        self.batch_sampler = batch_sampler
        if options is None:
            # Create IPU default options
            options = poptorch.Options()
        super().__init__(dataset=dataset,
                         num_nodes=num_nodes,
                         batch_size=batch_size,
                         batch_sampler=batch_sampler,
                         shuffle=shuffle,
                         exclude_keys=exclude_keys,
                         collater_args=collater_args,
                         options=options,
                         **kwargs)

    def _create_collater(self, **collater_args):
        base_collater = super()._create_collater(**collater_args)
        if self.batch_sampler is not None:
            mini_batch_size = None
        else:
            mini_batch_size = self.batch_size
        return CombinedBatchingCollater(mini_batch_size=mini_batch_size,
                                        collater=base_collater)


def create_fixed_batch_dataloader(
        dataset: Dataset,
        num_nodes: int,
        num_edges: Optional[int] = None,
        num_graphs: int = 1,
        options: Optional[poptorch.Options] = None,
        exclude_keys: Optional[Union[List[str], Tuple[str, ...]]] = None,
        collater_args: Optional[Dict[str, Union[int, float]]] = None,
        sampler: Optional[Union[Sampler[int], Iterable[int]]] = None,
        allow_skip_data: bool = False,
        **kwargs,
) -> FixedSizeDataLoader:
    r"""Creates a :class:`FixedSizeDataLoader` with :class:`FixedBatchSampler`
    for graph datasets.

    Args:
        dataset (Dataset): The :class:`~torch_geometric.data.Dataset` instance
            from which to load the graph examples for the IPU.
        num_nodes (int): Number of nodes in a batch.
        num_edges (int, optional): Number of edges in a batch.
            (default :obj:`None`)
        num_graphs (int, optional): How many graph examples to load in each
            batch. (default :obj:`1`)
        options (poptorch.Options, optional): The :class:`poptorch.Options`
            used by the :class:`poptorch.DataLoader`. Will use the default
            options if not provided. (default :obj:`None`)
        exclude_keys (list or tuple, optional): Keys to exclude from the
            batch. (default :obj:`None`)
        collater_args (dict, optional): The additional arguments passed to
            :class:`FixedSizeCollater`. They should not contain
            :obj:`num_nodes` or :obj:`exclude_keys` as those should be passed
            directly to the initializer method. (default :obj:`None`)
        sampler (Sampler or Iterable, optional): Base sampler. Can be any
            iterable object. (default :obj:`None`)
        allow_skip_data (bool, optional): Allow skip :obj:`data_source` item,
            otherwise throw :class:`RuntimeError` when the sampler is not able
            to form a single item batch from :obj:`data_source`, because
            :obj:`Data` exceeds the maximum batch requirements.
            (default :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`poptorch.DataLoader`.
    Returns:
        An instance of the :class:`FixedSizeDataLoader` class with the
        :class:`FixedBatchSampler` sampler.
    """
    return pyg_create_fixed_batch_dataloader(dataset,
                                             num_nodes=num_nodes,
                                             num_edges=num_edges,
                                             num_graphs=num_graphs,
                                             loader_cls=FixedSizeDataLoader,
                                             exclude_keys=exclude_keys,
                                             collater_args=collater_args,
                                             sampler=sampler,
                                             allow_skip_data=allow_skip_data,
                                             options=options,
                                             **kwargs)
