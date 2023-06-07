# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Note: The content of this file is going to be upstreamed to PyG.
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union

import torch.utils.data
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.data.data import BaseData

from poptorch_geometric.collate import FixedSizeCollater
from poptorch_geometric.fixed_size_options import FixedSizeOptions
from poptorch_geometric.pyg_collate import Collater
from poptorch_geometric.stream_packing_sampler import StreamPackingSampler


class FixedSizeStrategy(Enum):
    """Specify the strategy to use to achieve fixed-size mini-batches.

    - ``PadToMax``: Each mini-batch will contain a fixed number of real
      graphs plus one single graph for padding.
    - ``StreamPack``: If the next sample to batch can fit in the mini-batch
      it will be added. This results in mini-batches with a varied number
      of real graphs, but minimises the amount of wasted space in a
      mini-batch due to padding.
    """
    PadToMax = 0
    StreamPack = 1


class OverSizeStrategy(Enum):
    """Specify the behaviour if a sample cannot fit in the fixed-size
    mini-batch.

    - ``Error``:  If the required number of samples cannot fit into a
      mini-batch, an error will be thrown.
    - ``Skip``: If the required number of samples cannot fit into a
      mini-batch, the samples that cannot fit will be skipped.
    - ``TrimNodes``: If the required number of samples cannot fit into a
      mini-batch, the samples will still be added and then nodes will be
      removed from the mini-batch to achieve the fixed size. Enabling this
      can cause a loss of information in the samples of the mini-batch.
    - ``TrimEdges``: If the required number of samples cannot fit into a
      mini-batch, the samples will still be added and then edges will be
      removed from the mini-batch to achieve the fixed size. Enabling this
      can cause a loss of information in the samples of the mini-batch.
    - ``TrimNodesAndEdges``: If the required number of samples cannot fit
      into a mini-batch, the samples will still be added and then both
      nodes and edges will be removed from the mini-batch to achieve the
      fixed size. Enabling this can cause a loss of information in the
      samples of the mini-batch.
    """
    Error = 0
    Skip = 1
    TrimNodes = 2
    TrimEdges = 3
    TrimNodesAndEdges = 4


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


class FixedSizeDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from
    :class:`torch_geometric.data.Dataset` to a mini-batch and pads node and
    edge features so tensors across all batches have the same shapes.

    Data objects can be either of type :py:class:`~torch_geometric.data.Data` or
    :py:class:`~torch_geometric.data.HeteroData`.

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
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
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
            **kwargs,
    ) -> None:

        if fixed_size_options is None:
            self.fixed_size_options = FixedSizeOptions.from_dataset(
                dataset, batch_size)
        else:
            self.fixed_size_options = fixed_size_options

        if (isinstance(dataset[0], HeteroData)
                and not self.fixed_size_options.is_hetero()):
            self.fixed_size_options.to_hetero(dataset[0].node_types,
                                              dataset[0].edge_types)

        assert batch_size == self.fixed_size_options.num_graphs, (
            "`num_graphs` in fixed size options must match"
            " provided batch size in dataloader. `num_graphs`"
            f" is {self.fixed_size_options.num_graphs} but batch"
            f" size is {batch_size}.")
        self.padded_batch_size = batch_size

        batch_sampler = kwargs.pop("batch_sampler", None)

        if fixed_size_strategy == FixedSizeStrategy.StreamPack:
            if batch_sampler is not None:
                raise ValueError(
                    f"Fixed size strategy {fixed_size_strategy} is"
                    " incompatible with the provided batch_sampler"
                    f" {batch_sampler}. Either use a different strategy"
                    " or set `batch_sampler` to `None`.")
            base_sampler = RandomSampler(
                dataset) if shuffle else SequentialSampler(dataset)

            # Leave space for padding.
            sampler_graphs = batch_size - 1
            sampler_nodes = fixed_size_options.total_num_nodes - 1
            sampler_edges = fixed_size_options.total_num_edges - 1
            batch_sampler = StreamPackingSampler(
                dataset,
                sampler_graphs,
                max_num_nodes=sampler_nodes,
                max_num_edges=sampler_edges,
                base_sampler=base_sampler,
                allow_skip_data=(over_size_strategy == OverSizeStrategy.Skip))
        elif fixed_size_strategy != FixedSizeStrategy.PadToMax:
            raise NotImplementedError(
                f"Fixed size strategy {fixed_size_strategy} is not a supported"
                f" strategy for {self.__class__.__name__}")

        if batch_sampler is not None:
            # The `torch.DataLoader` class expects batch size to be `1`
            # and shuffle to be `None` when `batch_sampler` is provided.
            torch_dataloader_batch_size = 1
            shuffle = None
        else:
            torch_dataloader_batch_size = batch_size - 1

        self.batch_sampler = batch_sampler

        assert 'collate_fn' not in kwargs, \
            f'Cannot set `collate_fn` with `{self.__class__.__name__}`. ' \
            'Consider attaching a torch_geometric.transform.Pad transform' \
            ' after  your collate_fn and use with' \
            ' `torch.utils.dataloader.DataLoader`  to achieve fixed sized' \
            ' batches.'

        collater = self._create_collater(
            fixed_size_options=self.fixed_size_options,
            add_masks_to_batch=add_pad_masks,
            trim_nodes=(
                over_size_strategy in (OverSizeStrategy.TrimNodes,
                                       OverSizeStrategy.TrimNodesAndEdges)),
            trim_edges=(
                over_size_strategy in (OverSizeStrategy.TrimEdges,
                                       OverSizeStrategy.TrimNodesAndEdges)),
            follow_batch=follow_batch,
            exclude_keys=exclude_keys)

        super().__init__(dataset=dataset,
                         batch_size=torch_dataloader_batch_size,
                         shuffle=shuffle,
                         batch_sampler=batch_sampler,
                         collate_fn=collater,
                         **kwargs)

    def _create_collater(self, **collater_args):
        return FixedSizeCollater(**collater_args)
