# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.data.data import BaseData
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader.utils import get_input_nodes
from torch_geometric.sampler import NeighborSampler
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import EdgeType, InputNodes, OptTensor

import poptorch

from poptorch_geometric.collate import FixedSizeCollater
from poptorch_geometric.collate import CombinedBatchingCollater
from poptorch_geometric import OverSizeStrategy
from poptorch_geometric.fixed_size_options import FixedSizeOptions


class PyGFixedSizeNeighborLoader(torch.utils.data.DataLoader):
    def __init__(
            self,
            data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
            num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
            input_nodes: InputNodes = None,
            input_time: OptTensor = None,
            replace: bool = False,
            directed: bool = True,
            disjoint: bool = False,
            temporal_strategy: str = 'uniform',
            time_attr: Optional[str] = None,
            transform: Optional[Callable] = None,
            transform_sampler_output: Optional[Callable] = None,
            is_sorted: bool = False,
            filter_per_worker: bool = True,
            subgraph_type: SubgraphType = SubgraphType.directional,
            batch_size: int = 1,
            neighbor_sampler: Optional[NeighborSampler] = None,
            over_size_strategy: OverSizeStrategy = OverSizeStrategy.
            TrimNodesAndEdges,
            fixed_size_options: FixedSizeOptions = None,
            add_pad_masks: Optional[bool] = False,
            follow_batch: Optional[Union[List[str], Tuple[str, ...]]] = None,
            exclude_keys: Optional[Union[List[str], Tuple[str, ...]]] = None,
            options: Optional[poptorch.Options] = None,
            **kwargs,
    ):
        kwargs['batch_size'] = batch_size
        self.neighbour_loader = NeighborLoader(
            data,
            num_neighbors,
            input_nodes=input_nodes,
            input_time=input_time,
            replace=replace,
            subgraph_type=subgraph_type,
            directed=directed,
            disjoint=disjoint,
            temporal_strategy=temporal_strategy,
            time_attr=time_attr,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            is_sorted=is_sorted,
            filter_per_worker=filter_per_worker,
            neighbor_sampler=neighbor_sampler,
            **kwargs)
        self.input_type, input_nodes = get_input_nodes(data, input_nodes)

        if fixed_size_options is None:
            fixed_size_options = FixedSizeOptions.from_loader(
                self.neighbour_loader)

        collater_args = {}
        collater_args['fixed_size_options'] = fixed_size_options
        collater_args['add_masks_to_batch'] = add_pad_masks
        collater_args['follow_batch'] = follow_batch
        collater_args['exclude_keys'] = exclude_keys
        collater_args['trim_nodes'] = (over_size_strategy in (
            OverSizeStrategy.TrimNodes, OverSizeStrategy.TrimNodesAndEdges))
        collater_args['trim_edges'] = (over_size_strategy in (
            OverSizeStrategy.TrimEdges, OverSizeStrategy.TrimNodesAndEdges))

        kwargs['options'] = options
        collater = self._create_collater(**collater_args)
        super().__init__(dataset=range(input_nodes.size(0)),
                         collate_fn=collater,
                         **kwargs)

    def __collate__(self, index):
        out = self.nativeCollate(index)
        out = self.fixedSizeCollate(out)
        return out

    def _create_collater(self, **collater_args):
        self.fixed_size_collater = FixedSizeCollater(**collater_args)
        return self.__collate__

    def nativeCollate(self, index):
        out = self.neighbour_loader(index)
        return out

    def fixedSizeCollate(self, data_list: List[BaseData]):

        # Some keys are not handled correctly by FixedSizeCollater
        # so they need to be temporarily removed
        sample_batch_size = data_list[self.input_type].pop(
            "batch_size") if self.input_type else data_list.pop("batch_size")
        input_id = data_list[self.input_type].pop(
            "input_id") if self.input_type else data_list.pop("input_id")

        out = self.fixed_size_collater([data_list])

        # Restore previously removed keys
        if self.input_type:
            out[self.input_type].batch_size = sample_batch_size
            out[self.input_type].input_id = input_id
        else:
            out.batch_size = sample_batch_size
            out.input_id = input_id
        return out


class FixedSizeNeighborLoader(PyGFixedSizeNeighborLoader, poptorch.DataLoader):
    r"""A data loader which merges data objects from a
    :py:class:`torch_geometric.loader.NeighborLoader` to a mini-batch and pads
    node and edge features so tensors across all batches have constant shapes.

    Args:
        data (Any): A :class:`~torch_geometric.data.Data`,
            :class:`~torch_geometric.data.HeteroData`, or
            (:class:`~torch_geometric.data.FeatureStore`,
            :class:`~torch_geometric.data.GraphStore`) data object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbours to sample for each node in each iteration.
            If an entry is set to :obj:`-1`, all neighbors will be included.
            In heterogeneous graphs, it may also take a dictionary denoting
            the number of neighbours to sample for each individual edge type.
        input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
            indices of nodes for which neighbours are sampled to create
            mini-batches.
            Needs to be either given as a :obj:`torch.LongTensor` or
            :obj:`torch.BoolTensor`.
            If set to :obj:`None`, all nodes will be considered.
            In heterogeneous graphs, this needs to be passed as a tuple that
            holds the node type and node indices. (default: :obj:`None`)
        input_time (torch.Tensor, optional): Optional values to override the
            timestamp for the input nodes given in :obj:`input_nodes`. If not
            set, it will use the timestamps in :obj:`time_attr` as default (if
            present). The :obj:`time_attr` needs to be set for this to work.
            (default: :obj:`None`)
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        subgraph_type (SubgraphType or str, optional): The type of the returned
            subgraph.
            If set to :obj:`"directional"`, the returned subgraph only holds
            the sampled (directed) edges which are necessary to compute
            representations for the sampled seed nodes.
            If set to :obj:`"bidirectional"`, sampled edges are converted to
            bidirectional edges.
            If set to :obj:`"induced"`, the returned subgraph contains the
            induced subgraph of all sampled nodes.
            (default: :obj:`"directional"`)
        disjoint (bool, optional): If set to :obj:`True`, each seed node will
            create its own disjoint subgraph.
            If set to :obj:`True`, mini-batch outputs will have a :obj:`batch`
            vector holding the mapping of nodes to their respective subgraph.
            This will get automatically set to :obj:`True` in the case of
            temporal sampling. (default: :obj:`False`)
        temporal_strategy (str, optional): The sampling strategy when using
            temporal sampling (:obj:`"uniform"`, :obj:`"last"`).
            If set to :obj:`"uniform"`, it will sample uniformly across
            neighbours that fulfill temporal constraints.
            If set to :obj:`"last"`, will sample the last `num_neighbors` that
            fulfill temporal constraints.
            (default: :obj:`"uniform"`)
        time_attr (str, optional): The name of the attribute that denotes
            timestamps for the nodes in the graph.
             If set, temporal sampling will be used so that neighbours are
            guaranteed to fulfill temporal constraints; that is, neighbours have
            an earlier or equal timestamp than the centre node.
            (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        transform_sampler_output (callable, optional): A function/transform
            that takes in a :class:`torch_geometric.sampler.SamplerOutput` and
            returns a transformed version. (default: :obj:`None`)
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`edge_index` is sorted by column.
            If :obj:`time_attr` is set, additionally requires that rows are
            sorted by to time within individual neighbourhoods.
            This avoids internal re-sorting of the data and can improve
            runtime and memory efficiency. (default: :obj:`False`)
        filter_per_worker (bool, optional): This is left for argument
            compatibility with :obj:`NeighborLoader`. The passed value is
            ignored, FixedSizeNeighborLoader acts like filter_per_worker=True
        fixed_size_options (FixedSizeOptions, optional): A
            :py:class:`poptorch_geometric.fixed_size_options.FixedSizeOptions`
            object which holds the maximum number of nodes, edges and other
            options required to pad the mini-batches, produced by the data
            loader, to a fixed size.
        batch_size (int, optional): The number of nodes per mini-batch to
            load. (default: :obj:`1`)
        over_size_strategy (OverSizeStrategy, optional): The
            behaviour if a sample cannot fit in the fixed-size mini-batch.
            By default, if the required number of samples cannot fit into the
            fixed-sized mini-batch, nodes and edges will be removed from the
            mini-batch to achieve the specified fixed size.
            (default: `poptorch_geometric.OverSizeStrategy.TrimNodesAndEdges`)
        add_pad_masks (bool, optional): If :obj:`True`, mask objects
            are attached to mini-batch result. They represents three levels of
            padding:

            - :obj:`graphs_mask`: graph level mask
            - :obj:`nodes_mask`: node level mask
            - :obj:`edges_mask`: edge level mask

            Mask objects indicate which elements in the mini-batch are real
            (represented by :obj:`True`) and which were added as
            padding (represented by :obj:`False`).
            (default: :obj:`True`)
        options (poptorch.Options, optional): The additional PopTorch options
            to be passed to :py:class:`poptorch.DataLoader`.
            (default: :obj:`None`)
        exclude_keys (list or tuple, optional): The keys to exclude
            from the graphs in the output batch. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`shuffle`,
            :obj:`drop_last` or :obj:`num_workers`.

    """

    def __init__(
            self,
            data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
            num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
            input_nodes: InputNodes = None,
            input_time: OptTensor = None,
            subgraph_type: SubgraphType = SubgraphType.directional,
            replace: bool = False,
            directed: bool = True,
            disjoint: bool = False,
            temporal_strategy: str = 'uniform',
            time_attr: Optional[str] = None,
            transform: Optional[Callable] = None,
            transform_sampler_output: Optional[Callable] = None,
            is_sorted: bool = False,
            filter_per_worker: bool = True,  # Ignored
            batch_size: int = 1,
            neighbor_sampler: Optional[NeighborSampler] = None,
            over_size_strategy: OverSizeStrategy = OverSizeStrategy.
            TrimNodesAndEdges,
            fixed_size_options: FixedSizeOptions = None,
            add_pad_masks: Optional[bool] = True,
            follow_batch: Optional[Union[List[str], Tuple[str, ...]]] = None,
            exclude_keys: Optional[Union[List[str], Tuple[str, ...]]] = None,
            options: Optional[poptorch.Options] = None,
            **kwargs,
    ):
        self.batch_size = batch_size

        if options is None:
            # Create IPU default options
            options = poptorch.Options()

        super().__init__(
            data,
            num_neighbors,
            input_nodes=input_nodes,
            input_time=input_time,
            replace=replace,
            directed=directed,
            disjoint=disjoint,
            subgraph_type=subgraph_type,
            temporal_strategy=temporal_strategy,
            time_attr=time_attr,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            is_sorted=is_sorted,
            filter_per_worker=True,
            batch_size=batch_size,
            neighbor_sampler=neighbor_sampler,
            over_size_strategy=over_size_strategy,
            fixed_size_options=fixed_size_options,
            add_pad_masks=add_pad_masks,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
            options=options,
            **kwargs,
        )

    def _create_collater(self, **collater_args):
        collater = super()._create_collater(**collater_args)
        return CombinedBatchingCollater(mini_batch_size=self.batch_size,
                                        collater=collater)
