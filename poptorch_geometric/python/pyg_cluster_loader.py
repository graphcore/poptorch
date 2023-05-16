# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Dict, Optional, Union

import torch
from torch_geometric.loader import ClusterData, ClusterLoader

from poptorch_geometric.collate import FixedSizeCollater
from poptorch_geometric.fixed_size_options import FixedSizeOptions


class FixedSizeClusterLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.loader.ClusterData` to a mini-batch of clusters
    and pads node and edge features so tensors across all batches have constant
    shapes.

    Args:
        cluster_data (ClusterData): The cluster from which to load the data.
        fixed_size_options (FixedSizeOptions, optional): A
            :py:class:`poptorch_geometric.fixed_size_options.FixedSizeOptions`
            object which holds the maximum number of nodes, edges and other
            options required to pad the batches, produced by the data loader,
            to a fixed size.
        batch_size (int, optional): The number of samples per batch to load.
            (default: :obj:`1`)
        collater_args (dict, optional): The additional arguments passed to
            :class:`FixedSizeCollater`. They should not contain
            :obj:`num_nodes` or :obj:`exclude_keys` as those should be passed
            directly to the initializer method. (default: :obj:`None`)
        **kwargs (optional): The additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
            self,
            cluster_data: ClusterData,
            fixed_size_options: FixedSizeOptions,
            batch_size: int = 1,
            collater_args: Optional[Dict[str, Union[int, float]]] = None,
            **kwargs,
    ):
        assert fixed_size_options.num_graphs == 2, (
            "The number of graphs in a batch specified by the fixed sized"
            f" options must be 2 when using the {self.__class__.__name__},"
            " currently it is set to"
            f" {fixed_size_options.num_graphs}")

        unsupported = set(kwargs).intersection(
            {'collate_fn', 'batch_sampler', 'shuffle', 'exclude_keys'})
        assert not unsupported, \
            '`FixedSizeClusterLoader` does not support the following ' \
            f'arguments: {unsupported}.'

        collater_args = collater_args if collater_args else {}
        assert 'fixed_size_options' not in collater_args, \
            '`FixedSizeClusterLoader` uses argument `fixed_size_options`' \
            ' directly to the initializer. They should not be included' \
            ' in `collater_args`.'

        self.cluster_data = cluster_data
        self.batch_size = batch_size

        collater = self._create_collater(fixed_size_options=fixed_size_options,
                                         **collater_args)
        super().__init__(dataset=range(len(cluster_data)),
                         batch_size=batch_size,
                         collate_fn=collater,
                         **kwargs)

    def _collate(self, batch):
        batch = self.cluster_collater(batch)
        batch = self.fixed_size_collater([batch])
        return batch

    def _create_collater(self, **collater_args):
        cluster_loader = ClusterLoader(self.cluster_data,
                                       batch_size=self.batch_size)
        self.cluster_collater = cluster_loader._collate  # pylint: disable=protected-access
        self.fixed_size_collater = FixedSizeCollater(**collater_args)

        return self._collate
