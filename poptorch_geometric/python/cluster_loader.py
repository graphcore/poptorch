# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from __future__ import annotations  # noqa: F407

from typing import Optional

from torch_geometric.loader import ClusterData

from poptorch_geometric.collate import CombinedBatchingCollater
from poptorch_geometric.fixed_size_options import FixedSizeOptions
from poptorch_geometric.pyg_cluster_loader import \
    FixedSizeClusterLoader as PyGFixedSizeClusterLoader
from poptorch_geometric.pyg_dataloader import OverSizeStrategy
import poptorch


class FixedSizeClusterLoader(PyGFixedSizeClusterLoader, poptorch.DataLoader):
    r"""A data loader which merges data objects from a
    :py:class:`torch_geometric.loader.ClusterData` to a mini-batch of clusters
    and pads node and edge features so tensors across all batches have constant
    shapes.

    Args:
        cluster_data (ClusterData): The cluster from which to load the data.
        fixed_size_options (FixedSizeOptions, optional): A
            :py:class:`poptorch_geometric.fixed_size_options.FixedSizeOptions`
            object which holds the maximum number of nodes, edges and other
            options required to pad the mini-batches, produced by the data
            loader, to a fixed size.
        batch_size (int, optional): The number of nodes per mini-batch to
            load.
            (default: :obj:`1`)
        over_size_strategy (OverSizeStrategy, optional): The
            behaviour if a sample cannot fit in the fixed-size mini-batch.
            By default, if the required number of samples cannot fit into the
            fixed-sized mini-batch, nodes and edges will be removed from the
            mini-batch to achieve the specified fixed size.
            (default: `poptorch_geometric.OverSizeStrategy.TrimNodesAndEdges`)
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
        options (poptorch.Options, optional): The additional PopTorch options
            to be passed to :py:class:`poptorch.DataLoader`.
            (default: :obj:`None`)
        **kwargs (optional): The additional arguments of
            :py:class:`poptorch.DataLoader`.
    """

    def __init__(
            self,
            cluster_data: ClusterData,
            fixed_size_options: FixedSizeOptions,
            batch_size: int = 1,
            over_size_strategy: OverSizeStrategy = OverSizeStrategy.
            TrimNodesAndEdges,
            add_pad_masks: Optional[bool] = True,
            options: Optional[poptorch.Options] = None,
            **kwargs,
    ):
        self.batch_size = batch_size

        if options is None:
            # Create IPU default options
            options = poptorch.Options()

        super().__init__(cluster_data=cluster_data,
                         fixed_size_options=fixed_size_options,
                         batch_size=batch_size,
                         over_size_strategy=over_size_strategy,
                         add_pad_masks=add_pad_masks,
                         options=options,
                         **kwargs)

    def _create_collater(self, **collater_args):
        collater = super()._create_collater(**collater_args)
        return CombinedBatchingCollater(mini_batch_size=self.batch_size,
                                        collater=collater)
