# Copyright (c) 2022-2023 Graphcore Ltd. All rights reserved.
# Note: The content of this file is going to be upstreamed to PyG.
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import torch.utils.data
from torch.utils.data.sampler import Sampler
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData

from poptorch_geometric.batch_sampler import FixedBatchSampler
from poptorch_geometric.collate import FixedSizeCollater
from poptorch_geometric.pyg_collate import Collater


class TorchDataLoaderMeta(type):
    r"""Injects the :obj:`torch.utils.data.DataLoader` class as a base class
    of class that uses the :class:`TorchDataLoaderMeta` metaclass.

    This metaclass can be used when a dataloader needs to be aware of
    platform-specific features or limitations.
    In such case this metaclass allows to inject custom dataloader as a base
    class for :class:`FixedSizeDataLoader`. To achieve that, create a subclass
    of this metaclass, override its :obj:`base_loader` field and use that new
    class as a metaclass for subclass of :class:`FixedSizeDataLoader`.

    Example:

        # Creating a fixed size dataloader with CustomDataLoader as a base
        # class:

        >>> class CustomDataLoaderMeta(TorchDataLoaderMeta):
        ...     base_loader = CustomDataLoader

        >>> class CustomFixedSizeDataLoader(FixedSizeDataLoader,
        ...                                 metaclass=CustomDataLoaderMeta):
        ...     def __init__(self, *args, *kwargs):
        ...         super().__init__(*args, **kwargs)

    """
    base_loader = torch.utils.data.DataLoader

    def __call__(cls, *args, **kwargs):
        base_cls = cls.base_loader
        name = f"{cls.__name__}_{base_cls.__module__}.{base_cls.__name__}"

        class MetaResolver(type(cls), type(base_cls)):  # pylint: disable=duplicate-bases

            pass

        if name not in globals():
            globals()[name] = MetaResolver(name, (cls, base_cls), {})
        new_cls = globals()[name]

        return super(TorchDataLoaderMeta, new_cls).__call__(*args, **kwargs)


# ==== Copied from PyG and changed to use metaclass, have
# `_create_collater` method and pass arguments to `__init__`` as keyword ones.
class DataLoader(metaclass=TorchDataLoaderMeta):
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


class FixedSizeDataLoader(metaclass=TorchDataLoaderMeta):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch and pads node and
    edge features so tensors across all batches have constant shapes.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        num_nodes (int): The total number of nodes in the padded batch.
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
            num_nodes: int,
            batch_size: Optional[int] = None,
            batch_sampler: Optional[Sampler[List[int]]] = None,
            shuffle: bool = False,
            follow_batch: Optional[Union[List[str], Tuple[str, ...]]] = None,
            exclude_keys: Optional[Union[List[str], Tuple[str, ...]]] = None,
            collater_args: Optional[Dict[str, Union[int, float]]] = None,
            **kwargs,
    ):

        assert 'collate_fn' not in kwargs, \
            'Cannot set `collate_fn` with `FixedSizeDataLoader`. '\
            'Use `torch.utils.dataloader.DataLoader` directly if you need ' \
            'to specify collater manually.'

        collater_args = collater_args if collater_args else {}
        not_in_collater_args = {'num_nodes', 'follow_batch', 'exclude_keys'}
        invalid_collater_args = not_in_collater_args.intersection(
            set(collater_args))
        assert not invalid_collater_args, \
            '`FixedSizeDataLoader` uses arguments: ' \
            f'{", ".join(invalid_collater_args)} passed directly to the ' \
            'initializer. They should not be included in `collater_args`.'

        if batch_sampler is not None:
            self.padded_batch_size = batch_sampler.num_graphs + 1

            assert num_nodes >= batch_sampler.num_nodes + 1, \
                f'Argument `num_nodes` (= {num_nodes}) should be greater ' \
                f'than `num_nodes` (= {batch_sampler.num_nodes}) attribute ' \
                f'of `batch_sampler` in order to leave some space for padding.'

            # The `torch.DataLoader` class expects batch size to be `1` when
            # `batch_sampler` is provided.
            torch_dataloader_batch_size = 1
            if 'num_edges' not in collater_args:
                collater_args['num_edges'] = batch_sampler.num_edges + 1
        else:
            self.padded_batch_size = batch_size or 2
            num_real_graphs = self.padded_batch_size - 1
            torch_dataloader_batch_size = num_real_graphs

        if 'num_graphs' not in collater_args:
            collater_args['num_graphs'] = self.padded_batch_size

        collater = self._create_collater(num_nodes=num_nodes,
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


def create_fixed_batch_dataloader(
        dataset: Dataset,
        num_nodes: int,
        num_edges: Optional[int] = None,
        batch_size: int = 2,
        loader_cls: Type[FixedSizeDataLoader] = FixedSizeDataLoader,
        follow_batch: Optional[Union[List[str], Tuple[str, ...]]] = None,
        exclude_keys: Optional[Union[List[str], Tuple[str, ...]]] = None,
        collater_args: Optional[Dict[str, Union[int, float]]] = None,
        sampler: Optional[Union[Sampler[int], Iterable[int]]] = None,
        allow_skip_data: bool = False,
        **kwargs,
) -> FixedSizeDataLoader:
    r"""Creates a data loader based on the :obj:`loader_cls` class with
    :class:`FixedBatchSampler` for graph datasets.

    Args:
        dataset (Dataset): The :class:`~torch_geometric.data.Dataset` instance
            from which to load the graph examples.
        num_nodes (int): Number of nodes in a batch.
        num_edges (int, optional): Number of edges in a batch.
            (default: :obj:`None`)
        batch_size (int, optional): How many graph examples to load in each
            batch. This should be at least :obj:`2` to allow for creating at
            least one padding graph. (default: :obj:`2`)
        loader_cls (type, optional): Initialization class for the data loader.
            (default: :class:`FixedSizeDataLoader`)
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

    Returns:
        An instance of the :obj:`loader_cls` class with the
        :class:`FixedBatchSampler` sampler.
    """
    # Leave space for padding.
    sampler_graphs = batch_size - 1
    sampler_nodes = num_nodes - 1 if num_nodes is not None else num_nodes
    sampler_edges = num_edges - 1 if num_edges is not None else num_edges
    batch_sampler = FixedBatchSampler(dataset,
                                      sampler_graphs,
                                      num_nodes=sampler_nodes,
                                      num_edges=sampler_edges,
                                      sampler=sampler,
                                      allow_skip_data=allow_skip_data)

    return loader_cls(dataset=dataset,
                      num_nodes=num_nodes,
                      batch_sampler=batch_sampler,
                      follow_batch=follow_batch,
                      exclude_keys=exclude_keys,
                      collater_args=collater_args,
                      **kwargs)
