# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from torch_geometric.data import Batch
from torch_geometric.nn import avg_pool, voxel_grid
from torch_geometric.testing import withPackage

from pool_utils import pool_harness


@withPackage('torch_cluster')
def test_voxel_grid():
    pos = torch.Tensor([[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]])
    batch = torch.tensor([0, 0, 0, 1, 1])

    out = pool_harness(voxel_grid, [pos, 5, batch])
    assert out.tolist() == [0, 5, 3, 6, 7]
    out = pool_harness(voxel_grid, [pos, 5])
    assert out.tolist() == [0, 5, 3, 0, 1]


@pytest.mark.skip(reason="TODO(AFS-264, AFS-265)")
@withPackage('torch_cluster')
def test_voxel_grid_with_optional_args():
    pos = torch.Tensor([[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]])
    batch = torch.tensor([0, 0, 0, 1, 1])

    cluster = pool_harness(voxel_grid, [pos, 5, batch, -1, [18, 14]])
    assert cluster.tolist() == [0, 10, 4, 16, 17]

    cluster_no_batch = pool_harness(voxel_grid, [pos, 5, None, -1, [18, 14]])
    assert cluster_no_batch.tolist() == [0, 10, 4, 0, 1]


@withPackage('torch_cluster')
def test_single_voxel_grid():
    pos = torch.Tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    edge_index = torch.tensor([[0, 0, 3], [1, 2, 4]])
    batch = torch.tensor([0, 0, 0, 1, 1])
    x = torch.randn(5, 16)

    cluster = pool_harness(voxel_grid, [pos, 5, batch])
    assert cluster.tolist() == [0, 0, 0, 1, 1]

    data = Batch(x=x, edge_index=edge_index, pos=pos, batch=batch)
    data = avg_pool(cluster, data)

    cluster_no_batch = pool_harness(voxel_grid, [pos, 5])
    assert cluster_no_batch.tolist() == [0, 0, 0, 0, 0]
