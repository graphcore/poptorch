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
        return torch.full(self._shape, index, dtype=torch.float32)


class IncrementDatasetWithLabels(torch.utils.data.Dataset):
    def __init__(self, shape, length):
        self._shape = shape
        self._length = length

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return (torch.full(self._shape, index, dtype=torch.float32),
                torch.full((1, ), index, dtype=torch.long))


class CheckOrderModel(torch.nn.Module):
    def forward(self, data, expected):
        # return expected + 1 if data was what we expected
        return torch.sum(data - expected)


class DoubleData(torch.nn.Module):
    def forward(self, data):
        return data * 2


class DoubleDataLabel(torch.nn.Module):
    def forward(self, data, label):
        return data * 2, label * 2


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

    assert len(data) == num_tensors // (device_iterations * batch_size *
                                        replication_factor)
    model = poptorch.inferenceModel(CheckOrderModel(), opts)
    for it, d in enumerate(data):
        expected = torch.from_numpy(
            numpy.stack([
                numpy.full(shape, i, dtype=numpy.float32)
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


def _run_process_test(shape=None,
                      num_tensors=100,
                      batch_size=1,
                      num_workers=0,
                      device_iterations=1,
                      replication_factor=1,
                      num_runs=1):
    shape = shape or [2, 3]

    opts = poptorch.Options()
    opts.deviceIterations(device_iterations)
    opts.replicationFactor(replication_factor)

    data = poptorch.DataLoader(opts,
                               IncrementDataset(shape, num_tensors),
                               batch_size=batch_size,
                               num_workers=num_workers)

    loader = poptorch.AsynchronousDataAccessor(data)

    model = poptorch.inferenceModel(DoubleData(), opts)

    for _ in range(0, num_runs):
        for it, d in enumerate(loader):
            out = model(d)

            expected = torch.stack([
                torch.full(shape, i * 2, dtype=torch.float32)
                for i in range(data.combinedBatchSize *
                               it, data.combinedBatchSize * (it + 1))
            ])

            assert torch.equal(expected, out)


def test_multithreaded1():
    _run_process_test(num_tensors=100,
                      batch_size=2,
                      device_iterations=1,
                      replication_factor=1,
                      num_workers=0)


def test_multithreaded2():
    _run_process_test(num_tensors=100,
                      batch_size=2,
                      device_iterations=10,
                      replication_factor=1,
                      num_workers=0)


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed for replica > 1")
def test_multithreaded3():
    _run_process_test(num_tensors=10,
                      batch_size=2,
                      device_iterations=1,
                      replication_factor=4,
                      num_workers=0)


def _run_process_label_test(shape=None,
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
                               IncrementDatasetWithLabels(shape, num_tensors),
                               batch_size=batch_size,
                               num_workers=num_workers)

    loader = poptorch.AsynchronousDataAccessor(data)

    model = poptorch.inferenceModel(DoubleDataLabel(), opts)

    total = torch.zeros(shape)
    label_out = torch.zeros(1, dtype=torch.int)
    for _, (data, label) in enumerate(loader):
        out, label = model(data, label)
        total += torch.sum(out, dim=0)
        label_out += torch.sum(label, dim=0)

    actual = 0
    for i in range(0, num_tensors):
        actual += i * 2

    numpy.testing.assert_array_equal(total[0][0].numpy(), [actual])
    numpy.testing.assert_array_equal(label_out[0].item(), [actual])


def test_multithreaded4():
    _run_process_label_test(num_tensors=60,
                            batch_size=2,
                            device_iterations=10,
                            replication_factor=1,
                            num_workers=0)


def _run_dataset_test(shape=None,
                      num_tensors=100,
                      batch_size=1,
                      num_workers=0,
                      device_iterations=1,
                      replication_factor=1,
                      host_id=0,
                      num_hosts=1):
    shape = shape or [2, 3]

    opts = poptorch.Options()
    opts.deviceIterations(device_iterations)
    opts.replicationFactor(replication_factor)
    opts.Distributed.configureProcessId(host_id, num_hosts)

    data = poptorch.DataLoader(opts,
                               IncrementDataset(shape, num_tensors),
                               batch_size=batch_size,
                               num_workers=num_workers)

    offset = host_id * (num_tensors // num_hosts)
    assert len(data) == num_tensors // (device_iterations * batch_size *
                                        replication_factor * num_hosts)
    for it, d in enumerate(data):
        expected = torch.from_numpy(
            numpy.stack([
                numpy.full(shape, offset + i, dtype=numpy.float32)
                for i in range(data.combinedBatchSize *
                               it, data.combinedBatchSize * (it + 1))
            ]))
        diff = torch.sum(torch.sum(d - expected))

    numpy.testing.assert_array_equal(diff.numpy(), [0.])


def test_subdataset():
    _run_dataset_test(batch_size=4, host_id=0, num_hosts=2)


def test_subdataset2():
    _run_dataset_test(batch_size=2, host_id=1, num_hosts=2)


def test_interrupt_async_loader():
    """Make sure the worker processes are stopped cleanly even when the end of
    the dataset is not reached."""

    shape = [2, 3]
    num_tensors = 100

    opts = poptorch.Options()
    data = poptorch.DataLoader(opts,
                               IncrementDataset(shape, num_tensors),
                               batch_size=1,
                               num_workers=1)

    loader = poptorch.AsynchronousDataAccessor(data)
    for _, _ in enumerate(loader):
        break
