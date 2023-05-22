# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from __future__ import annotations  # noqa: F407

from typing import List, Optional, Tuple, Union

from torch_geometric.data import Dataset

import poptorch
from poptorch_geometric.collate import CombinedBatchingCollater
from poptorch_geometric.fixed_size_options import FixedSizeOptions
from poptorch_geometric.pyg_dataloader import DataLoader as PyGDataLoader
from poptorch_geometric.pyg_dataloader import FixedSizeDataLoader as PyGFixedSizeDataLoader
from poptorch_geometric.pyg_dataloader import FixedSizeStrategy, OverSizeStrategy


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


class FixedSizeDataLoader(PyGFixedSizeDataLoader, poptorch.DataLoader):
    r"""A data loader which merges data objects from
    :py:class:`poptorch.Dataset` into a mini-batch and pads node and
    edge features so tensors across all mini-batches have the same shapes.

    Data objects can be either of type :py:class:`~torch_geometric.data.Data`
    or :py:class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The :class:`~torch_geometric.data.Dataset` instance
            from which to load the graph samples.
        batch_size (int, optional): The number of graph samples to load in each
            mini-batch. This should be at least :obj:`2` to allow for creating
            at least one padding graph. (default: :obj:`2`)
        shuffle (bool, optional): If :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        fixed_size_options (FixedSizeOptions, optional): A
            :py:class:`poptorch_geometric.fixed_size_options.FixedSizeOptions`
            object which holds the maximum number of nodes, edges and other
            options required to pad the mini-batches, produced by the data
            loader, to a fixed size. If not specified, this will be determined
            from the provided dataset. (default: :obj:`None`)
        fixed_size_strategy (FixedSizeStrategy, optional): The
            strategy to use to achieve fixed-size mini-batches. By default,
            each mini-batch will contain a fixed number of real graphs
            (`batch_size` - 1) plus one single graph for padding.
            (default: `poptorch_geometric.FixedSizeStrategy.PadToMax`)
        over_size_strategy (OverSizeStrategy, optional): The
            behaviour if a sample cannot fit in the fixed-size mini-batch.
            By default, if the required number of samples cannot fit into the
            fixed-sized batch an error will be raised.
            (default: `poptorch_geometric.OverSizeStrategy.Error`)
        add_pad_masks  (bool, optional): If :obj:`True`, mask objects
            are attached to mini-batch result. They represents three levels of
            padding:

            - :obj:`graphs_mask` - graph level mask
            - :obj:`nodes_mask`  - node level mask
            - :obj:`edges_mask`  - edge level mask

            Mask objects indicate which elements in the mini-batch are real
            (represented by :obj:`True`) and which were added as
            padding (represented by :obj:`False`).
            (default: :obj:`True`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (list or tuple, optional): Keys to exclude from the
            batch. (default: :obj:`None`)
        options (poptorch.Options, optional): The additional PopTorch options
            to be passed to :py:class:`poptorch.DataLoader`.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :py:class:`poptorch.DataLoader`.
    """

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int = 2,
            shuffle: bool = False,
            fixed_size_options: Optional[FixedSizeOptions] = None,
            fixed_size_strategy: FixedSizeStrategy = FixedSizeStrategy.
            PadToMax,
            over_size_strategy: OverSizeStrategy = OverSizeStrategy.Error,
            add_pad_masks: Optional[bool] = True,
            follow_batch: Optional[Union[List[str], Tuple[str, ...]]] = None,
            exclude_keys: Optional[Union[List[str], Tuple[str, ...]]] = None,
            options: Optional[poptorch.Options] = None,
            **kwargs,
    ):
        if options is None:
            # Create IPU default options
            options = poptorch.Options()
        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         fixed_size_options=fixed_size_options,
                         fixed_size_strategy=fixed_size_strategy,
                         over_size_strategy=over_size_strategy,
                         add_pad_masks=add_pad_masks,
                         follow_batch=follow_batch,
                         exclude_keys=exclude_keys,
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
