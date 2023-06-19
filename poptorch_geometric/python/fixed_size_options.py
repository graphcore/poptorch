# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import DataLoader
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.data.summary import Summary
from torch_geometric.typing import EdgeType, NodeType


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
        num_nodes (int or dict): The number of nodes after
            padding a batch.
            In heterogeneous graphs, this can be a dictionary denoting
            the number of nodes for specific node types.
        num_edges (int or dict, optional): The number of edges after
            padding a batch.
            In heterogeneous graphs, this can be a dictionary denoting the
            number of edges for specific edge types.
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
                 num_nodes: Union[int, Dict[NodeType, int]],
                 num_edges: Optional[Union[int, Dict[EdgeType, int]]] = None,
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
            total_num_nodes = sum(self.num_nodes.values()) if isinstance(
                num_nodes, dict) else num_nodes
            self.num_edges = total_num_nodes * (total_num_nodes - 1)

        validate_num_graphs(num_graphs)
        self.num_graphs = num_graphs

        self.total_num_nodes_hetero = None
        self.total_num_edges_hetero = None

        self.node_pad_value = 0.0 if node_pad_value is None else node_pad_value
        self.edge_pad_value = 0.0 if edge_pad_value is None else edge_pad_value
        self.graph_pad_value = (0.0 if graph_pad_value is None else
                                graph_pad_value)
        self.pad_graph_defaults = ({} if pad_graph_defaults is None else
                                   pad_graph_defaults)

    def is_hetero(self):
        """Returns whether the specified number of nodes and edges are
        in heterogeneous form, ie a number for each node and edge type."""
        return (isinstance(self.num_nodes, dict)
                and isinstance(self.num_edges, dict))

    def to_hetero(self, node_types: List[NodeType],
                  edge_types: List[EdgeType]):
        """Converts a single specified number of nodes and edges to
        a heterogeneous form, a number for each node and edge type."""
        if not isinstance(self.num_nodes, dict):
            self.num_nodes = {k: self.num_nodes for k in node_types}
        if not isinstance(self.num_edges, dict):
            self.num_edges = {k: self.num_edges for k in edge_types}
        return self

    @property
    def total_num_nodes(self):
        """The total number of nodes summed for all the node types."""
        if self.is_hetero():
            if self.total_num_nodes_hetero is None:
                self.total_num_nodes_hetero = sum(self.num_nodes.values())
            return self.total_num_nodes_hetero
        return self.num_nodes

    @property
    def total_num_edges(self):
        """The total number of nodes summed for all the edge types."""
        if self.is_hetero():
            if self.total_num_edges_hetero is None:
                self.total_num_edges_hetero = sum(self.num_edges.values())
            return self.total_num_edges_hetero
        return self.num_edges

    @classmethod
    def from_dataset(cls,
                     dataset: Dataset,
                     batch_size: int,
                     sample_limit: Optional[int] = None,
                     progress_bar: Optional[bool] = None):
        """Returns a `FixedSizeOptions` object which is a valid set of
        options for the given dataset, ensuring that the number of nodes
        and edges allocated are enough for the dataset given a particular
        batch size."""

        validate_num_graphs(batch_size)

        if sample_limit is None:
            sample_limit = len(dataset)

        dataset_summary = Summary.from_dataset(dataset,
                                               progress_bar=progress_bar)

        def get_max_for_batch_size(batch_size, sample_max):
            return int(sample_max) * (batch_size - 1) + 1

        if dataset_summary.num_nodes_per_type:
            max_nodes_per_batch = {
                k: get_max_for_batch_size(batch_size, v.max)
                for k, v in dataset_summary.num_nodes_per_type.items()
            }
        else:
            max_nodes_per_batch = get_max_for_batch_size(
                batch_size, dataset_summary.num_nodes.max)

        if dataset_summary.num_edges_per_type:
            max_edges_per_batch = {
                k: get_max_for_batch_size(batch_size, v.max)
                for k, v in dataset_summary.num_edges_per_type.items()
            }
        else:
            max_edges_per_batch = get_max_for_batch_size(
                batch_size, dataset_summary.num_edges.max)

        return FixedSizeOptions(
            num_nodes=max_nodes_per_batch,
            num_edges=max_edges_per_batch,
            num_graphs=batch_size,
        )

    @classmethod
    def from_loader(cls, loader: DataLoader, sample_limit: int = 1000):
        """Returns a `FixedSizeOptions` object which is a valid set of
        options for the given data loader, ensuring that the number of nodes
        and edges allocated are approximately enough for the mini-batches
        produced by this data loader. As the underlying loader is unlikely
        to produce an exhaustive combination of samples in a mini-batch,
        the `FixedSizeOptions` returned can only be an approximation of the
        maximum values required."""

        is_hetero_data = isinstance(next(iter(loader)), HeteroData)

        max_num_graphs = 0
        max_num_nodes = dict() if is_hetero_data else 0
        max_num_edges = dict() if is_hetero_data else 0

        def loop_with_limit(loader, limit):
            count = 0
            while True:
                for sample in loader:
                    if count >= limit:
                        return
                    count += 1
                    yield sample

        for data in loop_with_limit(loader, sample_limit):
            if is_hetero_data:
                for node_type in data.node_types:
                    max_num_nodes[node_type] = max(
                        max_num_nodes.get(node_type, 0),
                        data[node_type].num_nodes)
                for edge_type in data.edge_types:
                    max_num_edges[edge_type] = max(
                        max_num_edges.get(edge_type, 0),
                        data[edge_type].num_edges)
            else:
                max_num_nodes = max(max_num_nodes, data.num_nodes)
                max_num_edges = max(max_num_edges, data.num_edges)

            if hasattr(data, "num_graphs"):
                max_num_graphs = max(max_num_graphs, data.num_graphs)
            else:
                max_num_graphs = 1

        # Allocate space for padding
        max_num_graphs += 1
        if is_hetero_data:
            max_num_nodes = {k: v + 1 for k, v in max_num_nodes.items()}
            max_num_edges = {k: v + 1 for k, v in max_num_edges.items()}
        else:
            max_num_nodes += 1
            max_num_edges += 1

        return FixedSizeOptions(
            num_nodes=max_num_nodes,
            num_edges=max_num_edges,
            num_graphs=max_num_graphs,
        )

    def __repr__(self):
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
