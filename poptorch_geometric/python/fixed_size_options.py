# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import itertools
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.data.summary import Summary


def validate_num_graphs(num_graphs):
    if num_graphs < 2:
        raise ValueError("The number of graphs in the batch must be"
                         " at least 2. This is to ensure the batch"
                         " contains at least 1 real graph and a graph"
                         " reserved for padding the batch to a fixed size.")


class FixedSizeOptions:
    r"""Class that holds the specification of how a data loader can be
    padded up to a fixed size. This includes the number of nodes and
    edges to pad a batch, produced using this specification, to a
    maximum number.

    Args:
        num_nodes (int): The total number of nodes in the
            padded batch.
        num_edges (int, optional): The total number of edges
            in the padded batch.
            (default: :obj:`num_nodes * (num_nodes - 1)`)
        num_graphs (int, optional): The total number of graphs
            in the padded batch. This should be at least :obj:`2` to allow
            for creating at least one padding graph. The default value is
            :obj:`2` accounting for a single real graph and a single padded
            graph in a batch.
            (default: :obj:`2`)
        node_pad_value (float, optional): The fill value to use for node
            features. (default: :obj:`0.0`)
        edge_pad_value (float, optional): The fill value to use for edge
            features. (default: :obj:`0.0`)
        graph_pad_value (float, optional): The fill value to use for graph
            features. (default: :obj:`0.0`)
        pad_graph_defaults (dict, optional): The default values that
            will be assigned to the keys of types different to
            :class:`torch.Tensor` in the newly created padding graphs.
            (default: :obj:`None`)
    """

    def __init__(self,
                 num_nodes: int,
                 num_edges: Optional[int] = None,
                 num_graphs: int = 2,
                 node_pad_value: Optional[float] = None,
                 edge_pad_value: Optional[float] = None,
                 graph_pad_value: Optional[float] = None,
                 pad_graph_defaults: Optional[Dict[str, Any]] = None):

        self.num_nodes = num_nodes

        if num_edges:
            self.num_edges = num_edges
        else:
            # Assume fully connected graph.
            self.num_edges = num_nodes * (num_nodes - 1)

        validate_num_graphs(num_graphs)
        self.num_graphs = num_graphs

        self.node_pad_value = 0.0 if node_pad_value is None else node_pad_value
        self.edge_pad_value = 0.0 if edge_pad_value is None else edge_pad_value
        self.graph_pad_value = (0.0 if graph_pad_value is None else
                                graph_pad_value)
        self.pad_graph_defaults = ({} if pad_graph_defaults is None else
                                   pad_graph_defaults)

    @classmethod
    def from_dataset(cls,
                     dataset: Dataset,
                     batch_size: int,
                     sample_limit: Optional[int] = None,
                     progress_bar: Optional[bool] = None):

        if isinstance(dataset[0], HeteroData):
            raise NotImplementedError(
                f"{cls.__class__.__name__}.from_dataset does not support"
                f" heterogeneous data. Instantiate {cls.__class__.__name__}"
                " directly.")

        validate_num_graphs(batch_size)

        if sample_limit is None:
            sample_limit = len(dataset)

        dataset_summary = Summary.from_dataset(dataset,
                                               progress_bar=progress_bar)

        max_nodes_per_batch = int(
            dataset_summary.num_nodes.max) * (batch_size - 1) + 1
        max_edges_per_batch = int(
            dataset_summary.num_edges.max) * (batch_size - 1) + 1

        return FixedSizeOptions(
            num_nodes=max_nodes_per_batch,
            num_edges=max_edges_per_batch,
            num_graphs=batch_size,
        )

    @classmethod
    def from_loader(cls, loader: DataLoader, sample_limit: int = 10000):

        if isinstance(next(iter(loader)), HeteroData):
            raise NotImplementedError(
                f"{cls.__class__.__name__}.from_loader does not support"
                f" heterogeneous data. Instantiate {cls.__class__.__name__}"
                " directly.")

        max_num_nodes, max_num_edges, max_num_graphs = 0, 0, 0
        samples = 0
        for data in itertools.cycle(loader):
            if data.num_nodes > max_num_nodes:
                max_num_nodes = data.num_nodes
            if data.num_edges > max_num_edges:
                max_num_edges = data.num_edges

            if data.batch is not None:
                num_graphs_in_batch = data.batch.max() + 1
            else:
                num_graphs_in_batch = 1

            if num_graphs_in_batch > max_num_graphs:
                max_num_graphs = num_graphs_in_batch

            if samples == sample_limit:
                break
            samples += 1

        return FixedSizeOptions(
            num_nodes=max_num_nodes + 1,
            num_edges=max_num_edges + 1,
            num_graphs=max_num_graphs + 1,
        )

    def __str__(self):
        return (f"{self.__class__.__name__}("
                f"num_nodes={self.num_nodes}"
                " (At least one node reserved for padding), "
                f"num_edges={self.num_edges}"
                " (At least one edge reserved for padding), "
                f"num_graphs={self.num_graphs}"
                " (At least one graph reserved for padding), "
                f"node_pad_value={self.node_pad_value}, "
                f"edge_pad_value={self.edge_pad_value}, "
                f"graph_pad_value={self.graph_pad_value}, "
                f"pad_graph_defaults={self.pad_graph_defaults})")
