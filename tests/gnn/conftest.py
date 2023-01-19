# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os.path as osp

import pytest
import torch_geometric as pyg


@pytest.fixture
def pyg_qm9():
    testdir = osp.abspath(osp.dirname(__file__))
    qm9root = osp.join(testdir, "..", "test_data", "qm9")
    return pyg.datasets.QM9(root=qm9root)


@pytest.fixture
def molecule(pyg_qm9):
    # The index of the largest molecule in the QM9 dataset, which looks like:
    # Data(edge_attr=[56, 4], edge_index=[2, 56], idx=[1], name="gdb_57518",
    #      pos=[29, 3], x=[29, 11], y=[1, 19], z=[29])
    max_index = 55967
    return pyg_qm9[max_index]


@pytest.fixture
def fake_hetero_dataset() -> pyg.datasets.FakeHeteroDataset:
    pyg.seed_everything(1410)
    dataset = pyg.datasets.FakeHeteroDataset(num_node_types=2,
                                             num_edge_types=5,
                                             avg_num_nodes=50)[0]
    return dataset


@pytest.fixture
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
