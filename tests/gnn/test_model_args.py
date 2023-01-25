# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.data import Batch
from torch_geometric.datasets import FakeDataset
from torch_geometric.nn.models import MLP

from utils import assert_equal
# Need to import poppyg to ensure that our arg parser implementation is
# registered with poptorch ahead of running these tests
import poppyg  # pylint: disable=unused-import
import poptorch


class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = MLP([in_channels, out_channels])

    def forward(self, example):
        example.h = self.mlp(example.x)
        example.out = F.log_softmax(example.h, dim=1)

        if self.training:
            pred = example.out[example.train_mask]
            target = example.y[example.train_mask]
            example.loss = F.cross_entropy(pred, target)

        return example


def add_train_mask(data):
    # Add a train_mask property that contains indices
    num_training_nodes = int(0.8 * data.num_nodes)
    data.train_mask = torch.randperm(data.num_nodes)[:num_training_nodes]
    return data


def data():
    seed_everything(0)
    dataset = FakeDataset(transform=add_train_mask,
                          avg_num_nodes=32,
                          num_channels=8)
    data = dataset[0]
    in_channels = data.x.shape[-1]
    out_channels = dataset.num_classes

    return data, in_channels, out_channels


def batch():
    seed_everything(0)
    dataset = FakeDataset(num_graphs=4,
                          transform=add_train_mask,
                          avg_num_nodes=12,
                          num_channels=8)
    data = dataset[0]
    in_channels = data.x.shape[-1]
    out_channels = dataset.num_classes
    batch = Batch.from_data_list(dataset[:])
    return batch, in_channels, out_channels


@pytest.fixture
def dispatcher_options():
    options = poptorch.Options()
    return options


@pytest.mark.parametrize('arg', [data(), batch()], ids=['data', 'batch'])
def test_args(arg, dispatcher_options):
    arg, in_channels, out_channels = arg

    if isinstance(arg, Batch):
        pytest.skip("Known issue. Unblock when AFS-97 will be completed.")

    model = Model(in_channels, out_channels)
    model.train()
    optimizer = poptorch.optim.Adam(model.parameters(), lr=0.001)
    model = poptorch.trainingModel(model=model,
                                   options=dispatcher_options,
                                   optimizer=optimizer)

    output = model(arg)
    assert isinstance(output, type(arg)), \
        "Model output must have the same type as input argument"

    # Check that all the keys from the input argument are also present on the
    # output argument.
    for k in arg.keys:
        assert k in output

    # Check that all the keys that were added in the model are present on the
    # output argument.
    for k in ['h', 'out', 'loss']:
        assert k in output

    if isinstance(arg, Batch):
        # Check that the batch vector is preserved but omit the dtype since
        # the PopTorch dispatcher will coerce long -> int32
        assert_equal(output.batch, arg.batch, check_dtype=False)
        assert output.batch.dtype == torch.int32
