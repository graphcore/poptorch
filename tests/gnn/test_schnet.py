# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import torch
import torch_geometric.nn.models.schnet as pygschnet

from torch.nn.functional import mse_loss
from torch.testing import assert_close
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose, Distance, RadiusGraph

from utils import assert_equal
from poptorch_geometric import Pad, SchNet, TrainingStepper, set_aggregation_dim_size
import poptorch

CUTOFF = 6.0
MAX_NUM_NODES = 30


def create_transform():
    def select_target(data):
        # The HOMO-LUMO gap is target 4 in QM9 dataset labels vector y
        target = 4
        data.y = data.y[0, target]
        return data

    keys = ("z", "edge_attr", "edge_index", "y", "num_nodes")

    return Compose([
        RadiusGraph(CUTOFF),
        Distance(norm=False, cat=False),
        Pad(max_num_nodes=MAX_NUM_NODES,
            edge_pad_value=CUTOFF,
            include_keys=keys),
        select_target,
    ])


@pytest.fixture(params=[1, 8])
def batch(pyg_qm9, request):
    batch_size = request.param
    pyg_qm9.transform = create_transform()
    data_list = list(pyg_qm9[0:batch_size])
    return Batch.from_data_list(data_list)


class InferenceHarness:
    def __init__(self,
                 batch_size,
                 mol_padding=0,
                 num_features=32,
                 num_gaussians=25,
                 num_interactions=2,
                 options=poptorch.Options()):
        super().__init__()
        self.seed = 0
        self.batch_size = batch_size
        self.mol_padding = mol_padding
        self.options = options
        self.create_model(num_features, num_gaussians, num_interactions)
        self.create_reference_model(num_features, num_gaussians,
                                    num_interactions)

    def create_model(self, num_features, num_gaussians, num_interactions):
        # Set seed before creating the model to ensure all parameters are
        # initialized to the same values as the PyG reference implementation.
        torch.manual_seed(self.seed)
        self.model = SchNet(num_features=num_features,
                            num_gaussians=num_gaussians,
                            num_interactions=num_interactions,
                            cutoff=CUTOFF,
                            batch_size=self.batch_size)
        self.model.eval()

    def create_reference_model(self, num_features, num_gaussians,
                               num_interactions):
        # Use PyG implementation as a reference implementation
        torch.manual_seed(0)
        self.ref_model = pygschnet.SchNet(hidden_channels=num_features,
                                          num_filters=num_features,
                                          num_gaussians=num_gaussians,
                                          num_interactions=num_interactions,
                                          cutoff=CUTOFF)

        self.ref_model.eval()

    def reference_output(self, batch):
        # Mask out fake atom data added as padding.
        real_atoms = batch.z > 0.

        if torch.all(~real_atoms):
            # All padding atoms case
            return torch.zeros(torch.max(batch.batch) + 1)

        out = self.ref_model(batch.z[real_atoms], batch.pos,
                             batch.batch[real_atoms])
        return out.view(-1)

    def compare(self, actual, batch):
        expected = self.reference_output(batch)

        if self.mol_padding == 0:
            assert_close(actual, expected)
            return

        pad_output = torch.zeros(self.mol_padding)
        assert_equal(actual[-self.mol_padding:], pad_output)
        assert_close(actual[:-self.mol_padding], expected)

    def test_cpu_padded(self, batch):
        # Run padded model on CPU and check the output agrees with the
        # reference implementation
        actual = self.model(batch.z, batch.edge_attr, batch.edge_index,
                            batch.batch)
        self.compare(actual, batch)

    def test_ipu(self, batch):
        pop_model = poptorch.inferenceModel(self.model, options=self.options)
        actual = pop_model(batch.z, batch.edge_attr, batch.edge_index,
                           batch.batch)
        self.compare(actual, batch)


def test_inference(batch):
    mol_padding = 2
    batch_size = torch.max(batch.batch).item() + 1 + mol_padding
    harness = InferenceHarness(batch_size, mol_padding)
    harness.test_cpu_padded(batch)
    harness.test_ipu(batch)


def test_training(batch):
    torch.manual_seed(0)
    batch_size = int(batch.batch.max()) + 1
    model = SchNet(num_features=32,
                   num_gaussians=25,
                   num_interactions=2,
                   batch_size=batch_size)
    model.train()
    set_aggregation_dim_size(model, MAX_NUM_NODES * batch_size)
    stepper = TrainingStepper(model, rtol=1e-4, atol=1e-4)

    num_steps = 40
    batch = (batch.z, batch.edge_attr, batch.edge_index, batch.batch, batch.y)
    stepper.run(num_steps, batch)


def test_loss():
    # Check that loss agrees with mse_loss implementation
    torch.manual_seed(0)
    input = torch.randn(10)
    target = torch.randn(10)
    actual = SchNet.loss(input, target)
    expected = mse_loss(input, target)
    assert_equal(actual, expected)

    # Insert random "padding" zeros.
    mask = torch.randn_like(input) > 0.6
    input[mask] = 0.0
    target[mask] = 0.0
    actual = SchNet.loss(input, target)
    expected = mse_loss(input[~mask], target[~mask])
    assert_equal(actual, expected)
