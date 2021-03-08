# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import time
import subprocess
import torch
import pytest
import numpy
import poptorch
import helpers


class BrokenDataset(torch.utils.data.Dataset):
    def __init__(self, length):
        super().__init__()
        self._length = length

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        assert False, "Broken dataset"


class IncrementDataset(torch.utils.data.Dataset):
    def __init__(self, shape, length):
        super().__init__()
        self._shape = shape
        self._length = length

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return torch.full(self._shape, index, dtype=torch.float32)


class IncrementIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, shape, length):
        super().__init__()
        self._shape = shape
        self._length = length

    def __iter__(self):
        for index in range(self._length):
            yield torch.full(self._shape, index, dtype=torch.float32)


class IncrementIterableDatasetWithLen(IncrementIterableDataset):
    def __len__(self):
        return self._length


class IncrementDatasetWithLabels(torch.utils.data.Dataset):
    def __init__(self, shape, length):
        super().__init__()
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

    loader = poptorch.DataLoader(opts,
                                 IncrementDataset(shape, num_tensors),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 mode=poptorch.DataLoaderMode.Async)

    assert len(loader) == num_tensors // (device_iterations * batch_size *
                                          replication_factor)

    model = poptorch.inferenceModel(DoubleData(), opts)

    for _ in range(0, num_runs):
        for it, d in enumerate(loader):
            out = model(d)

            expected = torch.stack([
                torch.full(shape, i * 2, dtype=torch.float32)
                for i in range(loader.combinedBatchSize *
                               it, loader.combinedBatchSize * (it + 1))
            ])

            helpers.assert_allequal(actual=out, expected=expected)


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

    loader = poptorch.DataLoader(opts,
                                 IncrementDatasetWithLabels(
                                     shape, num_tensors),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 mode=poptorch.DataLoaderMode.Async)

    assert len(loader) == num_tensors // (device_iterations * batch_size *
                                          replication_factor)

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

    loader = poptorch.DataLoader(opts,
                                 IncrementDataset(shape, num_tensors),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 mode=poptorch.DataLoaderMode.Async)

    offset = host_id * (num_tensors // num_hosts)
    assert len(loader) == num_tensors // (device_iterations * batch_size *
                                          replication_factor * num_hosts)
    for it, d in enumerate(loader):
        expected = torch.from_numpy(
            numpy.stack([
                numpy.full(shape, offset + i, dtype=numpy.float32)
                for i in range(loader.combinedBatchSize *
                               it, loader.combinedBatchSize * (it + 1))
            ]))
        diff = torch.sum(torch.sum(d - expected))

    numpy.testing.assert_array_equal(diff.numpy(), [0.])


def test_subdataset():
    _run_dataset_test(batch_size=4, host_id=0, num_hosts=2)


def test_subdataset2():
    _run_dataset_test(batch_size=2, host_id=1, num_hosts=2, num_workers=2)


def test_interrupt_async_loader():
    """Make sure the worker processes are stopped cleanly even when the end of
    the dataset is not reached."""

    shape = [2, 3]
    num_tensors = 100

    opts = poptorch.Options()
    loader = poptorch.DataLoader(opts,
                                 IncrementDataset(shape, num_tensors),
                                 batch_size=1,
                                 num_workers=1,
                                 mode=poptorch.DataLoaderMode.Async)

    assert len(loader) == num_tensors

    for _, _ in enumerate(loader):
        break


def test_single_epoch():
    shape = [2, 3]
    num_tensors = 100

    opts = poptorch.Options()
    loader = poptorch.DataLoader(opts,
                                 IncrementDataset(shape, num_tensors),
                                 batch_size=1,
                                 num_workers=32,
                                 mode=poptorch.DataLoaderMode.Async)

    assert len(loader) == num_tensors

    for _, _ in enumerate(loader):
        continue


def test_iterable_dataset():
    shape = [2, 3]
    num_tensors = 100

    loader = poptorch.AsynchronousDataAccessor(
        IncrementIterableDataset(shape, num_tensors))

    for _, _ in enumerate(loader):
        continue

    # Make sure it works for more than 1 epoch
    for _, _ in enumerate(loader):
        continue


def test_iterable_dataloader():
    shape = [2, 3]
    num_tensors = 100

    opts = poptorch.Options()
    loader = poptorch.DataLoader(opts,
                                 IncrementIterableDataset(shape, num_tensors),
                                 batch_size=1,
                                 num_workers=1,
                                 mode=poptorch.DataLoaderMode.Async)

    for _, t in enumerate(loader):
        assert t.shape == torch.Size([1, 2, 3])
        continue

    # Make sure it works for more than 1 epoch
    for _, _ in enumerate(loader):
        continue


@pytest.mark.parametrize("persistent_workers", {True, False})
def test_iterable_dataloader_reset(persistent_workers):
    shape = [2, 3]
    num_tensors = 10

    opts = poptorch.Options()
    loader = poptorch.DataLoader(opts,
                                 IncrementDataset(shape, num_tensors),
                                 persistent_workers=persistent_workers,
                                 batch_size=1,
                                 num_workers=1,
                                 mode=poptorch.DataLoaderMode.Async)

    # Interrupt the first iteration
    for i, t in enumerate(loader):
        assert t.shape == torch.Size([1, 2, 3])
        assert t[0][0][0] == i
        if i == 4:
            print(f"Last tensor first iteration {t}")
            break
        continue

    print("Second iterator")
    # Make sure the second iteration returns all the tensors
    for i, t in enumerate(loader):
        assert t[0][0][0] == i
    assert i == (num_tensors - 1)


def test_early_preload():
    shape = [2, 3]
    num_tensors = 10
    num_buffers = 5

    opts = poptorch.Options()
    data = IncrementDataset(shape, num_tensors)

    async_opts_preload = {'early_preload': True, 'buffer_size': num_buffers}
    async_opts_no_preload = {
        'early_preload': False,
        'buffer_size': num_buffers
    }
    dataloader_args = {
        'options': opts,
        'dataset': data,
        'batch_size': 1,
        'num_workers': 1
    }

    preload = poptorch.DataLoader(**dataloader_args,
                                  mode=poptorch.DataLoaderMode.Async,
                                  async_options=async_opts_preload)
    no_preload = poptorch.DataLoader(**dataloader_args,
                                     mode=poptorch.DataLoaderMode.Async,
                                     async_options=async_opts_no_preload)

    time.sleep(2)  # Give time for the worker to fill the buffer

    assert sum(no_preload._accessor._worker._ready_to_read_index) == 1  # pylint: disable=protected-access
    assert sum(preload._accessor._worker._ready_to_read_index) == num_buffers  # pylint: disable=protected-access


def test_batch_size_None():
    shape = [2, 3]
    num_tensors = 10

    opts = poptorch.Options()
    loader = poptorch.DataLoader(opts,
                                 IncrementIterableDataset(shape, num_tensors),
                                 batch_size=None,
                                 drop_last=False,
                                 num_workers=1,
                                 mode=poptorch.DataLoaderMode.Async)

    for _, t in enumerate(loader):
        assert t.shape == torch.Size([2, 3])
        continue

    # Make sure it works for more than 1 epoch
    for _, _ in enumerate(loader):
        continue


def test_len():
    shape = [2, 3]
    num_tensors = 10

    opts = poptorch.Options()
    loader = poptorch.DataLoader(opts,
                                 IncrementIterableDataset(shape, num_tensors),
                                 batch_size=None,
                                 drop_last=False,
                                 num_workers=1,
                                 mode=poptorch.DataLoaderMode.Async)

    with pytest.raises(TypeError,
                       match="'IncrementIterableDataset' has no len()"):
        len(loader)
    loader = poptorch.DataLoader(opts,
                                 IncrementIterableDatasetWithLen(
                                     shape, num_tensors),
                                 batch_size=None,
                                 drop_last=False,
                                 num_workers=1,
                                 mode=poptorch.DataLoaderMode.Async)

    len(loader)


def test_broken_dataset():
    num_tensors = 100

    opts = poptorch.Options()
    data = poptorch.DataLoader(opts,
                               BrokenDataset(num_tensors),
                               batch_size=1,
                               num_workers=32)

    with pytest.raises(RuntimeError, match="worker thread failed to start"):
        poptorch.AsynchronousDataAccessor(data)


def test_subprocess_async_loader():
    print(subprocess.check_output(
        ["python3", "-m", "pytest", __file__, "-k", "test_single_epoch"],
        stderr=subprocess.STDOUT).decode('utf-8'),
          flush=True)


def test_subprocess_broken_dataset():
    stdout = subprocess.check_output([
        "python3", "-m", "pytest", __file__, "-k", "test_broken_dataset", "-s"
    ],
                                     stderr=subprocess.STDOUT).decode('utf-8')
    print(stdout)
    assert "AssertionError: Broken dataset" in stdout, (
        "Couldn't find failure "
        "reason in stdout")


@pytest.mark.parametrize("DatasetType",
                         [IncrementDataset, IncrementIterableDataset])
def test_reuse_workers(DatasetType):
    shape = [2, 3]
    num_tensors = 10

    opts = poptorch.Options()
    loader = poptorch.DataLoader(opts,
                                 DatasetType(shape, num_tensors),
                                 batch_size=1,
                                 num_workers=2,
                                 mode=poptorch.DataLoaderMode.Async)
    loader_no_reuse = poptorch.DataLoader(opts,
                                          DatasetType(shape, num_tensors),
                                          batch_size=1,
                                          persistent_workers=False,
                                          num_workers=2,
                                          mode=poptorch.DataLoaderMode.Async)

    # Workers are created when the AsynchronousDataAccessor is instantiated
    # So the first iteration should be fast
    num_tensors = 0
    start = time.perf_counter()
    for _ in loader_no_reuse:
        num_tensors += 1
    end = time.perf_counter()
    print(f"First epoch no reuse: {end - start} {num_tensors}")

    # subsequent iterations will join and create new workers
    # when a new iterator is created.
    for _ in range(3):
        start = time.perf_counter()
        for _ in loader_no_reuse:
            num_tensors += 1
        end = time.perf_counter()
        print(f"Other epoch no reuse: {end - start}  {num_tensors}")

    start = time.perf_counter()
    num_tensors_reuse = 0
    for _ in loader:
        num_tensors_reuse += 1
    end = time.perf_counter()
    print(f"First epoch: {end - start} {num_tensors_reuse}")

    for _ in range(3):
        start = time.perf_counter()
        for _ in loader:
            num_tensors_reuse += 1
        end = time.perf_counter()
        print(f"Other epoch: {end - start} {num_tensors_reuse}")

    assert num_tensors_reuse == num_tensors
