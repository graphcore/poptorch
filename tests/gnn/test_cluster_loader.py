# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
from torch_geometric import seed_everything
from torch_geometric.datasets import FakeDataset
from torch_geometric.loader import ClusterData

from poptorch_geometric.cluster_loader import \
    FixedSizeClusterLoader as IPUFixedSizeClusterLoader
from poptorch_geometric.fixed_size_options import FixedSizeOptions
from poptorch_geometric.pyg_cluster_loader import FixedSizeClusterLoader
import poptorch


@pytest.mark.parametrize('loader_cls',
                         [FixedSizeClusterLoader, IPUFixedSizeClusterLoader])
@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('task', ['graph', 'node'])
def test_fixed_size_dataloader_with_cluster_data(loader_cls, batch_size,
                                                 benchmark, task):
    ipu_dataloader = loader_cls is IPUFixedSizeClusterLoader

    avg_degree = 3
    num_parts = 8
    seed_everything(42)

    dataset = FakeDataset(
        num_graphs=1,
        avg_num_nodes=128,
        avg_degree=avg_degree,
        num_channels=4,
        task=task,
    )[0]

    # Get a sensible value for the the maximum number of nodes.
    padded_num_nodes = dataset.num_nodes // num_parts * batch_size + 10
    padded_num_edges = (avg_degree + 5) * padded_num_nodes

    cluster_data = ClusterData(dataset, num_parts=num_parts, log=False)

    # Define the expected tensor sizes in the output.
    data = cluster_data.data
    data_attributes = (k for k, _ in data
                       if data.is_node_attr(k) or data.is_edge_attr(k))

    expected_sizes = {
        k:
        ((padded_num_nodes if data.is_node_attr(k) else padded_num_edges), 0)
        for k in data_attributes
    }
    # Special case for edge_index which is of shape [2, num_edges].
    expected_sizes['edge_index'] = (padded_num_edges, 1)

    # Special case for `y` being graph-lvl label
    if not data.is_node_attr('y'):
        expected_sizes['y'] = (2, 0)

    # Create a fixed size dataloader.
    kwargs = {
        'cluster_data':
        cluster_data,
        'fixed_size_options':
        FixedSizeOptions(num_nodes=padded_num_nodes,
                         num_edges=padded_num_edges,
                         num_graphs=2),
        'batch_size':
        batch_size
    }
    if ipu_dataloader:
        kwargs['options'] = poptorch.Options()

    loader = loader_cls(**kwargs)

    # Check that each batch matches the expected size.
    for batch in loader:
        sizes_match = all(
            getattr(batch, k).shape[dim] == size
            for k, (size, dim) in expected_sizes.items())
        assert sizes_match

    def loop():
        for _ in loader:
            pass

    benchmark(loop)
