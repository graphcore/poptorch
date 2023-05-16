# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Note: The content of this file is going to be upstreamed to PyG.
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch.utils.data
from torch.utils.data.sampler import Sampler
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData

from poptorch_geometric.collate import FixedSizeCollater
from poptorch_geometric.fixed_size_options import FixedSizeOptions
from poptorch_geometric.pyg_collate import Collater
from poptorch_geometric.stream_packing_sampler import StreamPackingSampler


# ==== Copied from PyG and changed to have `_create_collater` method and
# pass arguments to `__init__`` as keyword ones.
class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

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
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
            self,
            dataset: Union[Dataset, Sequence[BaseData]],
            batch_size: int = 1,
            shuffle: bool = False,
            follow_batch: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            **kwargs,
    ):

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        collater = self._create_collater(follow_batch=follow_batch,
                                         exclude_keys=exclude_keys)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collater,
            **kwargs,
        )

    def _create_collater(self, **collater_args):
        return Collater(**collater_args)


# ==== End of copied code


class CustomFixedSizeDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch and pads node and
    edge features so tensors across all batches have constant shapes.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        fixed_size_options (FixedSizeOptions, optional): A
            :py:class:`poptorch_geometric.fixed_size_options.FixedSizeOptions`
            object which holds the maximum number of nodes, edges and other
            options required to pad the batches, produced by the data loader,
            to a fixed size. If not specified, this will be determined from
            the provided dataset. (default: :obj:`None`)
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
            :class:`FixedSizeCollater`. They should not contain
            :obj:`num_nodes`, :obj:`follow_batch` and :obj:`exclude_keys` as
            those should be passed directly to the initializer method.
            (default: :obj:`None`)
        **kwargs (optional): The additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int = 2,
            fixed_size_options: Optional[int] = None,
            batch_sampler: Optional[Sampler[List[int]]] = None,
            shuffle: bool = False,
            follow_batch: Optional[Union[List[str], Tuple[str, ...]]] = None,
            exclude_keys: Optional[Union[List[str], Tuple[str, ...]]] = None,
            collater_args: Optional[Dict[str, Union[int, float]]] = None,
            **kwargs,
    ):
        if fixed_size_options is None:
            self.fixed_size_options = FixedSizeOptions.from_dataset(
                dataset, batch_size)
        else:
            self.fixed_size_options = fixed_size_options

        assert batch_size == self.fixed_size_options.num_graphs, (
            "`num_graphs` in fixed size options must match"
            " provided batch size in dataloader. `num_graphs`"
            f" is {self.fixed_size_options.num_graphs} but batch"
            f" size is {batch_size}.")

        assert 'collate_fn' not in kwargs, \
            'Cannot set `collate_fn` with `CustomFixedSizeDataLoader`. '\
            'Use `torch.utils.dataloader.DataLoader` directly if you need ' \
            'to specify collater manually.'

        collater_args = collater_args if collater_args else {}
        not_in_collater_args = {
            'fixed_size_options', 'follow_batch', 'exclude_keys'
        }
        invalid_collater_args = not_in_collater_args.intersection(
            set(collater_args))
        assert not invalid_collater_args, \
            '`CustomFixedSizeDataLoader` uses arguments: ' \
            f'{", ".join(invalid_collater_args)} passed directly to the ' \
            'initializer. They should not be included in `collater_args`.'

        self.padded_batch_size = batch_size
        if batch_sampler is not None:
            # The `torch.DataLoader` class expects batch size to be `1` when
            # `batch_sampler` is provided.
            torch_dataloader_batch_size = 1
        else:
            torch_dataloader_batch_size = batch_size - 1

        collater = self._create_collater(
            fixed_size_options=self.fixed_size_options,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
            **collater_args)

        super().__init__(dataset=dataset,
                         batch_size=torch_dataloader_batch_size,
                         batch_sampler=batch_sampler,
                         shuffle=shuffle,
                         collate_fn=collater,
                         **kwargs)

    def _create_collater(self, **collater_args):
        return FixedSizeCollater(**collater_args)


class FixedSizeDataLoader(CustomFixedSizeDataLoader):
    r"""A data loader which merges data objects from
    :class:`torch_geometric.data.Dataset` to a mini-batch and pads node and
    edge features so tensors across all batches have the same shapes.
    The data loader uses
    :py:class:`poptorch_geometric.stream_packing_sampler.StreamPackingSampler`
    to select the samples that will be batched together.

    If not specified, :obj:`fixed_size_options` will be retrieved from the
    dataset.

    Args:
        dataset (Dataset): The :class:`~torch_geometric.data.Dataset` instance
            from which to load the graph examples.
        fixed_size_options (FixedSizeOptions, optional): A
            :py:class:`poptorch_geometric.fixed_size_options.FixedSizeOptions`
            object which holds the maximum number of nodes, edges and other
            options required to pad the batches, produced by the data loader,
            to a fixed size. If not specified, this will be determined from
            the provided dataset. (default: :obj:`None`)
        batch_size (int, optional): How many graph examples to load in each
            batch. This should be at least :obj:`2` to allow for creating at
            least one padding graph. (default: :obj:`2`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (list or tuple, optional): Keys to exclude from the
            batch. (default: :obj:`None`)
        collater_args (dict, optional): The additional arguments passed to
            :class:`FixedSizeCollater`. They should not contain
            :obj:`num_nodes`, :obj:`follow_batch` and :obj:`exclude_keys` as
            those should be passed directly to the initializer method.
            (default: :obj:`None`)
        sampler (Sampler or Iterable, optional): Base sampler. Can be any
            iterable object. (default: :obj:`None`)
        allow_skip_data (bool, optional): Allow skip :obj:`data_source` item,
            otherwise throw :class:`RuntimeError` when the sampler is not able
            to form a single item batch from :obj:`data_source`, because
            :obj:`Data` exceeds the maximum batch requirements.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int = 2,
            fixed_size_options: Optional[FixedSizeOptions] = None,
            follow_batch: Optional[Union[List[str], Tuple[str, ...]]] = None,
            exclude_keys: Optional[Union[List[str], Tuple[str, ...]]] = None,
            collater_args: Optional[Dict[str, Union[int, float]]] = None,
            sampler: Optional[Union[Sampler[int], Iterable[int]]] = None,
            allow_skip_data: bool = False,
            **kwargs,
    ) -> None:

        if fixed_size_options is None:
            self.fixed_size_options = FixedSizeOptions.from_dataset(
                dataset, batch_size)
        else:
            self.fixed_size_options = fixed_size_options

        assert batch_size == self.fixed_size_options.num_graphs, (
            "`num_graphs` in fixed size options must match"
            " provided batch size in dataloader. `num_graphs`"
            f" is {self.fixed_size_options.num_graphs} but batch"
            f" size is {batch_size}.")

        # Leave space for padding.
        sampler_graphs = batch_size - 1
        sampler_nodes = self.fixed_size_options.num_nodes - 1
        sampler_edges = self.fixed_size_options.num_edges - 1
        batch_sampler = StreamPackingSampler(dataset,
                                             sampler_graphs,
                                             max_num_nodes=sampler_nodes,
                                             max_num_edges=sampler_edges,
                                             base_sampler=sampler,
                                             allow_skip_data=allow_skip_data)

        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         fixed_size_options=self.fixed_size_options,
                         batch_sampler=batch_sampler,
                         follow_batch=follow_batch,
                         exclude_keys=exclude_keys,
                         collater_args=collater_args,
                         **kwargs)
