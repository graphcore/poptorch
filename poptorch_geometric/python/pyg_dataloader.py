# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Note: The content of this file is going to be upstreamed to PyG.
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union

import torch.utils.data
from torch.utils.data.sampler import Sampler
from torch_geometric.data import Dataset

from poptorch_geometric.batch_sampler import FixedBatchSampler
from poptorch_geometric.collate import FixedSizeCollater


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
        **kwargs (optional): The additional arguments of
            :class:`torch.utils.data.DataLoader`.
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
            **kwargs,
    ):

        assert 'collate_fn' not in kwargs, \
            'Cannot set `collate_fn` with `FixedSizeDataLoader`. '\
            'Use `torch.utils.dataloader.DataLoader` directly if you need ' \
            'to specify collater manually.'

        collater_args = collater_args if collater_args else {}
        assert ('num_nodes' not in collater_args and
                'exclude_keys' not in collater_args), \
            'FixedSizeDataLoader uses arguments `num_nodes` and ' \
            '`exclude_keys` passed directly to the initializer. They should ' \
            'not be included in `collater_args`.'

        collater = self._create_collater(num_nodes=num_nodes,
                                         exclude_keys=exclude_keys,
                                         **collater_args)
        super().__init__(dataset=dataset,
                         batch_size=batch_size,
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
        num_graphs: int = 1,
        loader_cls: Type[FixedSizeDataLoader] = FixedSizeDataLoader,
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
            (default :obj:`None`)
        num_graphs (int, optional): How many graph examples to load in each
            batch. (default :obj:`1`)
        loader_cls (type, optional): Initialization class for the data loader.
            (default :class:`FixedSizeDataLoader`)
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
            :class:`torch.utils.data.DataLoader`.

    Returns:
        An instance of the :obj:`loader_cls` class with the
        :class:`FixedBatchSampler` sampler.
    """
    batch_sampler = FixedBatchSampler(dataset,
                                      num_graphs,
                                      num_nodes=num_nodes,
                                      num_edges=num_edges,
                                      sampler=sampler,
                                      allow_skip_data=allow_skip_data)

    return loader_cls(dataset=dataset,
                      num_nodes=num_nodes,
                      batch_sampler=batch_sampler,
                      exclude_keys=exclude_keys,
                      collater_args=collater_args,
                      **kwargs)
