# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
from torch_geometric import seed_everything
from torch_geometric.datasets import FakeDataset
from torch_geometric.transforms import NormalizeFeatures

from poptorch_geometric.dataloader import FixedSizeDataLoader
from poptorch_geometric.fixed_size_options import FixedSizeOptions


@pytest.fixture
def dataloader():
    seed_everything(42)

    dataset = FakeDataset(num_graphs=4,
                          avg_num_nodes=8,
                          avg_degree=3,
                          transform=NormalizeFeatures(),
                          num_channels=8)

    dataloader = FixedSizeDataLoader(
        dataset,
        fixed_size_options=FixedSizeOptions(num_nodes=12, num_edges=32),
        collater_args={'add_masks_to_batch': True})

    return dataloader
