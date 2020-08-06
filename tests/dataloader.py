# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy
import poptorch
import pytest
import torch


class IncrementDataset(torch.utils.data.Dataset):
    def __init__(self, shape, length):
        self._shape = shape
        self._length = length

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return torch.nn.init.constant_(torch.Tensor(*self._shape), index)


class CheckOrderModel(torch.nn.Module):
    def forward(self, data, expected):
        # return expected + 1 if data was what we expected
        return torch.sum(data - expected)


def _run_test(shape=None,
              num_tensors=100,
              batch_size=1,
              num_workers=0,
              device_iterations=1,
              replication_factor=1):
    shape = shape or [2, 3]

    opts = poptorch.Options()
    opts.deviceIterations(device_iterations)
    opts.replicationFactor(replication_factor)

    data = poptorch.DataLoader(opts,
                               IncrementDataset(shape, num_tensors),
                               batch_size=batch_size,
                               num_workers=num_workers)

    model = poptorch.inferenceModel(CheckOrderModel(), opts)
    for it, d in enumerate(data):
        expected = torch.from_numpy(
            numpy.stack([
                numpy.full(shape, i, dtype=float)
                for i in range(data.combinedBatchSize *
                               it, data.combinedBatchSize * (it + 1))
            ]))
        diff = torch.sum(model(d, expected))

    numpy.testing.assert_array_equal(diff.numpy(), [0.])


def test_simple():
    _run_test()


def test_batch():
    _run_test(batch_size=4)


def test_workers():
    _run_test(num_workers=8)


def test_device_iterations():
    _run_test(device_iterations=4)


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed for replica > 1")
def test_replica():
    _run_test(replication_factor=4)


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed for replica > 1")
def test_combined():
    _run_test(batch_size=2,
              device_iterations=5,
              replication_factor=2,
              num_workers=4)
