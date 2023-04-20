# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os.path as osp
import pytest
import torch_geometric as pyg


@pytest.fixture(scope="module")
def pyg_qm9(pytestconfig):
    qm9root = osp.join(pytestconfig.getoption("external_datasets_dir"), "qm9")
    if not osp.exists(qm9root):
        raise RuntimeError(f'Path {qm9root} not exists.')
    return pyg.datasets.QM9(root=qm9root)


@pytest.fixture(scope="module")
def planetoid_cora(pytestconfig):
    planetoid_root = osp.join(pytestconfig.getoption("external_datasets_dir"),
                              "planetoid")
    if not osp.exists(planetoid_root):
        raise RuntimeError(f'Path {planetoid_root} not exists.')
    return pyg.datasets.Planetoid(planetoid_root,
                                  "Cora",
                                  transform=pyg.transforms.NormalizeFeatures())


@pytest.fixture(scope="module")
def molecule(pyg_qm9):
    # The index of the largest molecule in the QM9 dataset, which looks like:
    # Data(edge_attr=[56, 4], edge_index=[2, 56], idx=[1], name="gdb_57518",
    #      pos=[29, 3], x=[29, 11], y=[1, 19], z=[29])
    max_index = 55967
    return pyg_qm9[max_index]


@pytest.fixture(scope="module")
def fake_small_dataset() -> pyg.datasets.FakeDataset:
    pyg.seed_everything(42)
    dataset = pyg.datasets.FakeDataset(num_graphs=10,
                                       avg_num_nodes=30,
                                       avg_degree=5)
    return dataset


@pytest.fixture(scope="module")
def fake_large_dataset() -> pyg.datasets.FakeDataset:
    pyg.seed_everything(42)
    dataset = pyg.datasets.FakeDataset(num_graphs=100, avg_num_nodes=10)
    return dataset


@pytest.fixture(scope="module")
def fake_node_task_dataset() -> pyg.datasets.FakeDataset:
    pyg.seed_everything(42)
    dataset = pyg.datasets.FakeDataset(num_graphs=500,
                                       avg_num_nodes=10,
                                       task='node')
    return dataset


@pytest.fixture(scope="module")
def fake_hetero_dataset() -> pyg.datasets.FakeHeteroDataset:
    pyg.seed_everything(1410)
    dataset = pyg.datasets.FakeHeteroDataset(num_graphs=100,
                                             num_node_types=2,
                                             num_edge_types=5,
                                             avg_num_nodes=50)
    return dataset


@pytest.fixture(scope="module")
def fake_node_task_hetero_dataset() -> pyg.datasets.FakeHeteroDataset:
    pyg.seed_everything(1410)
    dataset = pyg.datasets.FakeHeteroDataset(num_graphs=100,
                                             num_node_types=2,
                                             num_edge_types=5,
                                             avg_num_nodes=50,
                                             task='node')
    return dataset


@pytest.fixture(scope="module")
def fake_hetero_data() -> pyg.datasets.FakeHeteroDataset:
    pyg.seed_everything(1410)
    dataset = pyg.datasets.FakeHeteroDataset(num_graphs=1,
                                             num_node_types=2,
                                             num_edge_types=5,
                                             avg_num_nodes=50)
    return dataset[0]


@pytest.fixture(scope="module")
def fake_molecular_dataset() -> pyg.datasets.FakeDataset:
    # setup a dataset which looks like a molecular dataset.
    pyg.seed_everything(42)
    avg_num_nodes = 20
    avg_degree = 3
    dataset = pyg.datasets.FakeDataset(
        num_graphs=1000,
        avg_num_nodes=avg_num_nodes,
        avg_degree=avg_degree,
        num_channels=20,
        task="graph",
    )
    return dataset
