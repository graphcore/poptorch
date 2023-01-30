# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import unittest.mock

import pytest
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.datasets import FakeDataset
from torch_geometric.nn.models import GAT, GCN, GIN, PNA, EdgeCNN, GraphSAGE
from torch_geometric.transforms import Compose, GCNNorm, NormalizeFeatures
from torch_geometric.utils import degree
from torch_scatter import scatter_add

import helpers
from poptorch_geometric import TrainingStepper, set_aggregation_dim_size


@pytest.fixture
def data():
    seed_everything(0)
    transform = Compose([GCNNorm(), NormalizeFeatures()])
    dataset = FakeDataset(transform=transform, num_channels=64)
    data = dataset[0]
    data.num_classes = dataset.num_classes

    # Add a train_mask property that contains indices
    num_training_nodes = int(0.8 * data.num_nodes)
    data.train_mask = torch.randperm(data.num_nodes)[:num_training_nodes]
    return data


def node_classification_harness(gnn,
                                dataset,
                                num_steps=40,
                                atol=1e-4,
                                rtol=1e-5):
    # Wrapper for a GNN model + a loss function
    class Wrapper(torch.nn.Module):
        def __init__(self, model, loss_fn):
            super().__init__()
            self.model = model
            self.loss_fn = loss_fn

        def forward(self, x, edge_index, train_mask, y):
            x = self.model(x, edge_index)
            out = F.log_softmax(x, dim=1)
            pred = out[train_mask]
            target = y[train_mask]
            loss = self.loss_fn(pred, target)
            return out, loss

    set_aggregation_dim_size(gnn, int(dataset.edge_index.max()) + 1)
    model = Wrapper(gnn, F.cross_entropy)
    stepper = TrainingStepper(model, atol=atol, rtol=rtol)
    batch = (dataset.x, dataset.edge_index, dataset.train_mask, dataset.y)
    stepper.run(num_steps, batch)


@pytest.mark.skip(reason="Known issue. Unblock when AFS-88 will be completed.")
def test_node_classification_GCN(data):
    gnn = GCN(in_channels=data.num_node_features,
              hidden_channels=32,
              num_layers=2,
              out_channels=data.num_classes,
              normalize=False)

    node_classification_harness(gnn, data)


@pytest.mark.skip(reason="Known issue. Unblock when AFS-88 will be completed.")
def test_node_classification_GraphSAGE(data):
    gnn = GraphSAGE(in_channels=data.num_node_features,
                    hidden_channels=32,
                    num_layers=2,
                    out_channels=data.num_classes)

    node_classification_harness(gnn, data, atol=1e-3, rtol=1e-2)


@pytest.mark.skip(reason="Known issue. Unblock when AFS-88 will be completed.")
def test_node_classification_GIN(data):
    gnn = GIN(in_channels=data.num_node_features,
              hidden_channels=32,
              num_layers=2,
              out_channels=data.num_classes)

    node_classification_harness(gnn, data)


@pytest.mark.skip(reason="Known issue. Unblock when AFS-88 will be completed.")
def test_node_classification_GAT(data):
    gnn = GAT(in_channels=data.num_node_features,
              hidden_channels=32,
              num_layers=2,
              out_channels=data.num_classes,
              add_self_loops=False)

    node_classification_harness(gnn, data)


@pytest.mark.skip(reason="Known issue. Unblock when AFS-88 will be completed.")
def test_node_classification_PNA(data):
    # Calculate the in-degree histogram
    deg = degree(data.edge_index[1]).long()
    deg = scatter_add(torch.ones_like(deg), deg)

    gnn = PNA(in_channels=data.num_node_features,
              hidden_channels=32,
              num_layers=2,
              out_channels=data.num_classes,
              aggregators=['sum', 'mean'],
              scalers=['linear'],
              deg=deg)

    # TODO: investigate numerical drift with PNAConv
    node_classification_harness(gnn, data, num_steps=1)


@unittest.mock.patch.dict("os.environ", helpers.disableSmallModel())
@pytest.mark.parametrize('act', [torch.nn.ReLU(), torch.relu_])
def test_node_classification_EdgeCNN(data, act):
    if act == torch.relu_:
        # TODO: enable testing with the inplace relu_ op when this is supported
        pytest.skip(
            "Skipping testing inplace activation with dispatcher: "
            "RuntimeError: a leaf Variable that requires grad is being used in"
            "an in-place operation.")

    gnn = EdgeCNN(in_channels=data.num_node_features,
                  hidden_channels=32,
                  num_layers=2,
                  out_channels=data.num_classes,
                  dropout=0,
                  act=act,
                  norm=None,
                  jk=None)

    node_classification_harness(gnn, data, num_steps=1)
