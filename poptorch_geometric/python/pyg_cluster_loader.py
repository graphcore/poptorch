# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Dict, Optional, Union

from torch_geometric.loader import ClusterData, ClusterLoader

from poptorch_geometric.collate import FixedSizeCollater
from poptorch_geometric.pyg_dataloader import TorchDataLoaderMeta


class FixedSizeClusterLoader(metaclass=TorchDataLoaderMeta):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.loader.ClusterData` to a mini-batch of clusters
    and pads node and edge features so tensors across all batches have constant
    shapes.

    Args:
        cluster_data (ClusterData): The cluster from which to load the data.
        num_nodes (int): The total number of nodes in the padded batch.
        batch_size (int, optional): The number of samples per batch to load.
            (default: :obj:`1`)
        collater_args (dict, optional): The additional arguments passed to
            :class:`FixedSizeCollater`. They should not contain
            :obj:`num_nodes` or :obj:`exclude_keys` as those should be passed
            directly to the initializer method. (default :obj:`None`)
        **kwargs (optional): The additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
            self,
            cluster_data: ClusterData,
            num_nodes: int,
            batch_size: int = 1,
            collater_args: Optional[Dict[str, Union[int, float]]] = None,
            **kwargs,
    ):
        unsupported = set(kwargs).intersection(
            {'collate_fn', 'batch_sampler', 'shuffle', 'exclude_keys'})
        assert not unsupported, \
            '`FixedSizeClusterLoader` does not support the following ' \
            f'arguments: {unsupported}.'

        collater_args = collater_args if collater_args else {}
        assert 'num_nodes' not in collater_args, \
            '`FixedSizeDataLoader` uses argument `num_nodes` passed directly' \
            'to the initializer. They should not be included' \
            'in `collater_args`.'

        self.cluster_data = cluster_data
        self.batch_size = batch_size

        collater = self._create_collater(num_nodes=num_nodes, **collater_args)
        super().__init__(dataset=range(len(cluster_data)),
                         batch_size=batch_size,
                         collate_fn=collater,
                         **kwargs)

    def __collate__(self, batch):
        batch = self.cluster_collater(batch)
        batch = self.fixed_size_collater([batch])
        return batch

    def _create_collater(self, **collater_args):
        cluster_loader = ClusterLoader(self.cluster_data,
                                       batch_size=self.batch_size)
        self.cluster_collater = cluster_loader.__collate__
        self.fixed_size_collater = FixedSizeCollater(**collater_args)

        return self.__collate__
