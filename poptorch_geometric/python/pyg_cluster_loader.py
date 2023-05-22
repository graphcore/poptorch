# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Optional

import torch
from torch_geometric.loader import ClusterData, ClusterLoader

from poptorch_geometric.collate import FixedSizeCollater
from poptorch_geometric.fixed_size_options import FixedSizeOptions
from poptorch_geometric.pyg_dataloader import OverSizeStrategy


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
        **kwargs (optional): The additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
            self,
            cluster_data: ClusterData,
            fixed_size_options: FixedSizeOptions,
            batch_size: int = 1,
            over_size_strategy: OverSizeStrategy = OverSizeStrategy.
            TrimNodesAndEdges,
            add_pad_masks: Optional[bool] = True,
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

        self.cluster_data = cluster_data
        self.batch_size = batch_size

        collater = self._create_collater(
            fixed_size_options=fixed_size_options,
            add_masks_to_batch=add_pad_masks,
            trim_nodes=(
                over_size_strategy in (OverSizeStrategy.TrimNodes,
                                       OverSizeStrategy.TrimNodesAndEdges)),
            trim_edges=(
                over_size_strategy in (OverSizeStrategy.TrimEdges,
                                       OverSizeStrategy.TrimNodesAndEdges)))

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
