# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest

from torch_geometric import seed_everything
from torch_geometric.datasets import FakeDataset
from torch_geometric.transforms import Compose, GCNNorm, NormalizeFeatures


def get_dataset(num_channels=16):
    seed_everything(0)
    transform = Compose([GCNNorm(), NormalizeFeatures()])

    dataset = FakeDataset(avg_num_nodes=32,
                          avg_degree=5,
                          transform=transform,
                          num_channels=num_channels)
    data = dataset[0]
    data.num_classes = dataset.num_classes

    return data


@pytest.fixture
def dataset():
    return get_dataset()
