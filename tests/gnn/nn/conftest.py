# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest

from torch_geometric import seed_everything
from torch_geometric.datasets import FakeDataset
from torch_geometric.transforms import Compose, GCNNorm, NormalizeFeatures

from poptorch_geometric.dataloader import FixedSizeDataLoader, DataLoader
from poptorch_geometric.fixed_size_options import FixedSizeOptions
from poptorch_geometric.pyg_dataloader import FixedSizeStrategy


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


@pytest.fixture
def fake_dataset():
    seed_everything(0)

    dataset = FakeDataset(num_graphs=4,
                          avg_num_nodes=8,
                          avg_degree=3,
                          transform=NormalizeFeatures(),
                          num_channels=10)
    return dataset


@pytest.fixture
def fixed_size_dataloader(fake_dataset):
    dataloader = FixedSizeDataLoader(
        fake_dataset,
        fixed_size_options=FixedSizeOptions(num_nodes=12),
        fixed_size_strategy=FixedSizeStrategy.StreamPack,
        add_pad_masks=True)
    return dataloader


@pytest.fixture
def dataloader(fake_dataset):
    dataloader = DataLoader(fake_dataset, shuffle=False)
    return dataloader
