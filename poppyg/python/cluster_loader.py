# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from __future__ import annotations  # noqa: F407

from typing import Dict, Optional, Union

from torch_geometric.loader import ClusterData

from poppyg.collate import CombinedBatchingCollater
from poppyg.dataloader import PopTorchDataLoaderMeta
from poppyg.pyg_cluster_loader import \
    FixedSizeClusterLoader as PyGFixedSizeClusterLoader
import poptorch


class FixedSizeClusterLoader(PyGFixedSizeClusterLoader,
                             metaclass=PopTorchDataLoaderMeta):
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
            :obj:`num_nodes` as it should be passed directly to the initializer
            method. (default :obj:`None`)
        options (poptorch.Options, optional): The additional PopTorch options
            to be passed to :obj:`poptorch.DataLoader`. (default: :obj:`None`)
        **kwargs (optional): The additional arguments of
            :class:`poptorch.DataLoader`.
    """

    def __init__(
            self,
            cluster_data: ClusterData,
            num_nodes: int,
            batch_size: int = 1,
            collater_args: Optional[Dict[str, Union[int, float]]] = None,
            options: Optional[poptorch.Options] = None,
            **kwargs,
    ):
        self.batch_size = batch_size

        if options is None:
            # Create IPU default options
            options = poptorch.Options()

        super().__init__(cluster_data=cluster_data,
                         num_nodes=num_nodes,
                         batch_size=batch_size,
                         collater_args=collater_args,
                         options=options,
                         **kwargs)

    def _create_collater(self, **collater_args):
        collater = super()._create_collater(**collater_args)
        return CombinedBatchingCollater(mini_batch_size=self.batch_size,
                                        collater=collater)
