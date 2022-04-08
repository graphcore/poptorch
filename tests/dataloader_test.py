# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import itertools
import math
import time
import subprocess
import marshal
import re
import os
import sys
import signal
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
    def __init__(self, shape, length, dtype=torch.float32):
        super().__init__()
        self._shape = shape
        self._length = length
        self._dtype = dtype

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index >= self._length:
            raise StopIteration
        return torch.full(self._shape, index, dtype=self._dtype)


class IncrementIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, shape, length, start=0, dtype=torch.float32):
        super().__init__()
        self._shape = shape
        self.length = length
        self.start = start
        self._dtype = dtype

    def __iter__(self):
        for index in range(self.length):
            yield torch.full(self._shape,
                             self.start + index,
                             dtype=self._dtype)


class IncrementIterableDatasetWithLen(IncrementIterableDataset):
    def __len__(self):
        return self.length


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


class IncrementDatasetWithLabelsDict(torch.utils.data.Dataset):
    def __init__(self, shape, length):
        super().__init__()
        self._shape = shape
        self._length = length

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return {
            "data": torch.full(self._shape, index, dtype=torch.float32),
            "label": torch.full((1, ), index, dtype=torch.long)
        }


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


def _run_test(trace_model,
              shape=None,
              num_tensors=100,
              batch_size=1,
              num_workers=0,
              device_iterations=1,
              replication_factor=1):
    shape = shape or [2, 3]

    opts = poptorch.Options()
    opts.deviceIterations(device_iterations)
    opts.replicationFactor(replication_factor)
    opts.Jit.traceModel(trace_model)

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


@pytest.mark.parametrize("trace_model", [True, False])
def test_simple(trace_model):
    _run_test(trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_batch(trace_model):
    _run_test(trace_model, batch_size=4)


@pytest.mark.parametrize("trace_model", [True, False])
def test_workers(trace_model):
    _run_test(trace_model, num_workers=8)


@pytest.mark.parametrize("trace_model", [True, False])
def test_device_iterations(trace_model):
    _run_test(trace_model, device_iterations=4)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_replica(trace_model):
    _run_test(trace_model, replication_factor=4)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_combined(trace_model):
    _run_test(trace_model,
              batch_size=2,
              device_iterations=5,
              replication_factor=2,
              num_workers=4)


def _run_process_test(trace_model,
                      shape=None,
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
    opts.Jit.traceModel(trace_model)

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


@pytest.mark.parametrize("trace_model", [True, False])
def test_multithreaded1(trace_model):
    _run_process_test(trace_model,
                      num_tensors=100,
                      batch_size=2,
                      device_iterations=1,
                      replication_factor=1,
                      num_workers=0)


@pytest.mark.parametrize("trace_model", [True, False])
def test_multithreaded2(trace_model):
    _run_process_test(trace_model,
                      num_tensors=100,
                      batch_size=2,
                      device_iterations=10,
                      replication_factor=1,
                      num_workers=0)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_multithreaded3(trace_model):
    _run_process_test(trace_model,
                      num_tensors=10,
                      batch_size=2,
                      device_iterations=1,
                      replication_factor=4,
                      num_workers=0)


def _run_process_label_test(trace_model,
                            shape=None,
                            num_tensors=100,
                            batch_size=1,
                            num_workers=0,
                            device_iterations=1,
                            replication_factor=1):
    shape = shape or [2, 3]

    opts = poptorch.Options()
    opts.deviceIterations(device_iterations)
    opts.replicationFactor(replication_factor)
    opts.Jit.traceModel(trace_model)

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
    for (data, label) in loader:
        out, label = model(data, label)
        total += torch.sum(out, dim=0)
        label_out += torch.sum(label, dim=0)

    expected = 0
    for i in range(0, num_tensors):
        expected += i * 2

    numpy.testing.assert_array_equal(total[0][0].numpy(), [expected])
    numpy.testing.assert_array_equal(label_out[0].item(), [expected])


@pytest.mark.parametrize("trace_model", [True, False])
def test_multithreaded4(trace_model):
    _run_process_label_test(trace_model,
                            num_tensors=60,
                            batch_size=2,
                            device_iterations=10,
                            replication_factor=1,
                            num_workers=0)


def _run_subdataset_test(num_tensors=100,
                         batch_size=1,
                         num_workers=0,
                         device_iterations=1,
                         replication_factor=1,
                         num_hosts=1):
    shape = [2, 3]
    dataset = IncrementDataset(shape, num_tensors)

    combined_batch_size = 0
    next_expected = 0
    for host_id in range(num_hosts):
        opts = poptorch.Options()
        opts.deviceIterations(device_iterations)
        opts.replicationFactor(replication_factor)
        opts.Distributed.configureProcessId(host_id, num_hosts)

        loader = poptorch.DataLoader(opts,
                                     dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     mode=poptorch.DataLoaderMode.Async)

        combined_batch_size = loader.combinedBatchSize
        assert combined_batch_size == (device_iterations * batch_size *
                                       replication_factor)
        assert len(loader) == num_tensors // (combined_batch_size * num_hosts)
        for d in loader:
            for elt in d:
                val = int(elt[0][0].item())
                assert val == next_expected
                next_expected += 1

    # Number of processes shouldn't change how many tensors are returned
    num_expected = num_hosts * combined_batch_size * (
        num_tensors // (combined_batch_size * num_hosts))
    assert next_expected == num_expected


def _run_shuffle_subdataset_test(num_tensors=100,
                                 batch_size=1,
                                 num_workers=0,
                                 device_iterations=1,
                                 replication_factor=1,
                                 num_hosts=1):
    shape = [2, 3]
    dataset = IncrementDataset(shape, num_tensors)

    total = [False] * num_tensors
    for host_id in range(num_hosts):
        seen = [False] * num_tensors
        opts = poptorch.Options()
        opts.deviceIterations(device_iterations)
        opts.replicationFactor(replication_factor)
        opts.randomSeed(42)
        opts.Distributed.configureProcessId(host_id, num_hosts)

        loader = poptorch.DataLoader(opts,
                                     dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     mode=poptorch.DataLoaderMode.Async)

        combined_batch_size = loader.combinedBatchSize
        assert combined_batch_size == (device_iterations * batch_size *
                                       replication_factor)
        assert len(loader) == num_tensors // (combined_batch_size * num_hosts)
        for d in loader:
            for elt in d:
                val = int(elt[0][0].item())
                assert not seen[val]
                seen[val] = True
                total[val] = True
        assert seen.count(
            True) == combined_batch_size * (num_tensors //
                                            (combined_batch_size * num_hosts))

        # Iterate a second time to make sure the left over tensors get used too.
        for d in loader:
            for elt in d:
                val = int(elt[0][0].item())
                total[val] = True

    # If we iterate twice in all the processes then all the tensors should be used.
    assert total.count(True) == num_tensors


def test_subdataset():
    _run_subdataset_test(batch_size=4, num_hosts=3)


def test_subdataset2():
    _run_subdataset_test(batch_size=2, num_hosts=2, num_workers=3)


def test_shuffle_subdataset():
    _run_shuffle_subdataset_test(batch_size=4, num_hosts=3)


def test_shuffle_subdataset2():
    _run_shuffle_subdataset_test(batch_size=2, num_hosts=2, num_workers=3)


@pytest.mark.parametrize("num_processes", [2, 3, 4, 5])
@pytest.mark.parametrize("num_workers", [0, 1, 3])
def test_global_shuffle_each_epoch(num_processes, num_workers):
    each_process_data = []
    for process_id in range(num_processes):
        each_process_data.append(list())
        opts = poptorch.Options()
        opts.randomSeed(0)
        opts.Distributed.configureProcessId(process_id, num_processes)
        dataloader = poptorch.DataLoader(
            opts,
            IncrementDataset((), 100),
            batch_size=16,
            shuffle=True,
            num_workers=num_workers,
        )
        for _ in range(5):
            each_epoch_data = []
            for batch in dataloader:
                each_epoch_data += batch.tolist()
            each_process_data[process_id].append(sorted(each_epoch_data))

    # Make sure data between epochs differs within the same process
    # for all processes.
    for process_data in each_process_data:
        for epoch_data_i, epoch_data_j in itertools.combinations(
                process_data, 2):
            assert epoch_data_i != epoch_data_j


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

    assert sum(no_preload._accessor._worker._data_buffers.indices_mem) == 1  # pylint: disable=protected-access, no-member
    assert sum(
        preload._accessor._worker._data_buffers.indices_mem) == num_buffers  # pylint: disable=protected-access, no-member


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


def test_iterable_dataset_len():
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

    with pytest.raises(poptorch.Error, match="worker thread failed to start"):
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
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_reuse_workers(DatasetType, dtype):
    shape = [2, 3]
    num_tensors = 10

    opts = poptorch.Options()
    loader = poptorch.DataLoader(opts,
                                 DatasetType(shape, num_tensors, dtype=dtype),
                                 batch_size=1,
                                 num_workers=2,
                                 mode=poptorch.DataLoaderMode.Async)
    loader_no_reuse = poptorch.DataLoader(opts,
                                          DatasetType(shape,
                                                      num_tensors,
                                                      dtype=dtype),
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


# Select a subset of the dataset for each worker
def _worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    total_len = dataset.length
    per_worker = math.ceil(dataset.length / worker_info.num_workers)
    dataset.start = per_worker * worker_id
    if worker_id == worker_info.num_workers - 1:
        dataset.length = total_len - (per_worker *
                                      (worker_info.num_workers - 1))
    else:
        dataset.length = per_worker


@pytest.mark.parametrize(
    "mode", {
        poptorch.DataLoaderMode.Async, poptorch.DataLoaderMode.AsyncRebatched,
        poptorch.DataLoaderMode.Sync
    })
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_iterable_dataloader_drop_last(mode, dtype):
    shape = [2, 3]
    num_tensors = 101
    num_workers = 7
    batch_size = 4
    if mode != poptorch.DataLoaderMode.AsyncRebatched:
        # Expected tensors
        # tensors per worker = ceil(101/7) = 15
        # last worker = 10 tensor
        # batch size = 4
        # Total = 6 * floor(15 / 4) + floor(10/4)
        #       = 6 * 3 + 2 = 20
        # Unused tensors = 101 - num_expected * 4 = 21
        num_expected = 20 * batch_size
    else:
        # Best case expected: floor(101/4) = 25 -> unused = 1
        num_expected = math.floor(num_tensors / batch_size) * batch_size

    opts = poptorch.Options()
    loader = poptorch.DataLoader(opts,
                                 IncrementIterableDataset(shape,
                                                          num_tensors,
                                                          dtype=dtype),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 mode=mode,
                                 drop_last=True,
                                 worker_init_fn=_worker_init_fn)

    values = set()
    for t in loader:
        assert t.shape == torch.Size([4, 2, 3])
        for b in t:
            v = int(b[0][0])
            assert v not in values
            values.add(v)

    assert len(values) == num_expected
    print("Missing tensors:")
    for i in range(num_tensors):
        if i not in values:
            print(i)

    # Make sure it works for more than 1 epoch
    values = set()
    for t in loader:
        assert t.shape == torch.Size([4, 2, 3])
        for b in t:
            v = int(b[0][0])
            assert v not in values
            values.add(v)

    assert len(values) == num_expected


@pytest.mark.parametrize(
    "mode", {
        poptorch.DataLoaderMode.Async, poptorch.DataLoaderMode.AsyncRebatched,
        poptorch.DataLoaderMode.Sync
    })
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_indexable_dataloader_drop_last(mode, dtype):
    shape = [2, 3]
    num_tensors = 101
    num_workers = 7
    batch_size = 4
    # Expected tensors
    # Best case expected: floor(101/4) = 25 -> unused = 1
    num_expected = 100

    opts = poptorch.Options()
    loader = poptorch.DataLoader(opts,
                                 IncrementDataset(shape,
                                                  num_tensors,
                                                  dtype=dtype),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 mode=mode)

    values = set()
    for t in loader:
        assert t.shape == torch.Size([4, 2, 3])
        for b in t:
            v = int(b[0][0])
            assert v not in values
            values.add(v)

    assert len(values) == num_expected
    print("Missing tensors:")
    for i in range(num_tensors):
        if i not in values:
            print(i)

    # Make sure it works for more than 1 epoch
    values = set()
    for t in loader:
        assert t.shape == torch.Size([4, 2, 3])
        for b in t:
            v = int(b[0][0])
            assert v not in values
            values.add(v)

    assert len(values) == num_expected


@pytest.mark.parametrize(
    "mode", {
        poptorch.DataLoaderMode.Async, poptorch.DataLoaderMode.AsyncRebatched,
        poptorch.DataLoaderMode.Sync
    })
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_indexable_dataloader_len(mode, dtype):
    shape = [2, 3]
    num_tensors = 101
    num_workers = 7
    batch_size = 4
    ds = IncrementDataset(shape, num_tensors, dtype=dtype)
    assert len(ds) == num_tensors
    n = 0
    for n, _ in enumerate(ds):
        pass
    assert n + 1 == num_tensors
    opts = poptorch.Options()
    loader = poptorch.DataLoader(opts,
                                 ds,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 mode=mode)
    if mode == poptorch.DataLoaderMode.Sync:
        # Make sure the user can still manually create the
        # data accessor. (This can only be tested in Sync
        # mode as otherwise the loader already contains
        # a data accessor).
        accessor = poptorch.AsynchronousDataAccessor(loader)
        assert len(loader) == num_tensors // batch_size
        for n, _ in enumerate(accessor):
            pass
        assert n + 1 == num_tensors // batch_size
        accessor = poptorch.AsynchronousDataAccessor(ds)
        assert len(accessor) == num_tensors
        for n, _ in enumerate(accessor):
            pass
        assert n + 1 == num_tensors

    assert len(loader) == num_tensors // batch_size
    for n, _ in enumerate(loader):
        pass
    assert n + 1 == num_tensors // batch_size


@pytest.mark.parametrize(
    "mode", {
        poptorch.DataLoaderMode.Async, poptorch.DataLoaderMode.AsyncRebatched,
        poptorch.DataLoaderMode.Sync
    })
def test_dictionary_dataset(mode):
    shape = [2, 3]
    num_tensors = 500

    opts = poptorch.Options()
    opts.deviceIterations(2)
    opts.replicationFactor(3)

    loader = poptorch.DataLoader(opts,
                                 IncrementDatasetWithLabelsDict(
                                     shape, num_tensors),
                                 num_workers=3,
                                 mode=mode)
    shape_with_batch = [loader.combinedBatchSize] + shape
    it = 0
    for d in loader:
        assert isinstance(d, dict)
        assert len(d) == 2
        assert "data" in d
        assert "label" in d
        assert d["data"].shape == torch.Size(shape_with_batch)
        assert d["label"].shape == torch.Size([loader.combinedBatchSize, 1])
        it += 1

    assert it == num_tensors // loader.combinedBatchSize


@pytest.mark.parametrize(
    "mode", {
        poptorch.DataLoaderMode.Async, poptorch.DataLoaderMode.AsyncRebatched,
        poptorch.DataLoaderMode.Sync
    })
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_iterable_dataloader_len(mode, dtype):
    shape = [2, 3]
    num_tensors = 101
    num_workers = 7
    batch_size = 4
    # Note: Upstream torch returns the theoretical length
    # it doesn't take into account the items lost per worker.
    expected_len = math.floor(num_tensors / batch_size)
    if mode != poptorch.DataLoaderMode.AsyncRebatched:
        # Expected tensors
        # tensors per worker = ceil(101/7) = 15
        # last worker = 10 tensor
        # batch size = 4
        # Total = 6 * floor(15 / 4) + floor(10/4)
        #       = 6 * 3 + 2 = 20
        # Unused tensors = 101 - num_iterations_expected * 4 = 21
        num_iterations_expected = 20
    else:
        # Best case expected: floor(101/4) = 25 -> unused = 1
        num_iterations_expected = expected_len
    ds = IncrementIterableDatasetWithLen(shape, num_tensors, dtype=dtype)
    assert len(ds) == num_tensors
    n = 0
    for n, _ in enumerate(ds):
        pass
    assert n + 1 == num_tensors
    opts = poptorch.Options()
    loader = poptorch.DataLoader(opts,
                                 ds,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 worker_init_fn=_worker_init_fn,
                                 mode=mode)
    if mode == poptorch.DataLoaderMode.Sync:
        accessor = poptorch.AsynchronousDataAccessor(loader)
        assert len(loader) == expected_len
        for n, _ in enumerate(accessor):
            pass
        assert n + 1 == num_iterations_expected
        accessor = poptorch.AsynchronousDataAccessor(ds)
        assert len(accessor) == num_tensors
        for n, _ in enumerate(accessor):
            pass
        assert n + 1 == num_tensors

    assert len(loader) == expected_len
    for n, _ in enumerate(loader):
        pass
    assert n + 1 == num_iterations_expected


@pytest.mark.parametrize(
    "mode",
    {poptorch.DataLoaderMode.AsyncRebatched, poptorch.DataLoaderMode.Sync})
@pytest.mark.parametrize("DatasetType",
                         [IncrementDataset, IncrementIterableDatasetWithLen])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_leftover(mode, DatasetType, dtype):
    shape = [2, 3]
    num_tensors = 101
    num_workers = 7
    batch_size = 6
    # Note: Upstream torch returns the theoretical length
    # it doesn't take into account the items lost per worker.
    expected_len = math.ceil(num_tensors / batch_size)

    ds = DatasetType(shape, num_tensors, dtype=dtype)
    if isinstance(ds, torch.utils.data.IterableDataset
                  ) and mode != poptorch.DataLoaderMode.AsyncRebatched:
        # Expected tensors
        # tensors per worker = ceil(101/7) = 15
        # last worker = 11 tensor
        # batch size = 6
        # Total = 6 * floor(15 / 6) + floor(11/6)
        #       = 6 * 2 + 1 = 13
        # Left over per worker: 3, 5 for the first one
        num_full_iterations_expected = 13
        left_over_batches = [5] + [3] * 6
    else:
        # Best case expected: floor(101/6) = 16 -> unused = 5
        num_full_iterations_expected = 16
        left_over_batches = [5]
    assert len(ds) == num_tensors
    n = 0
    for n, d in enumerate(ds):
        assert d.shape == torch.Size(shape)

    assert n + 1 == num_tensors
    opts = poptorch.Options()
    worker_init_fn = None
    if isinstance(ds, torch.utils.data.IterableDataset):
        worker_init_fn = _worker_init_fn
    loader = poptorch.DataLoader(opts,
                                 ds,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 worker_init_fn=worker_init_fn,
                                 drop_last=False,
                                 mode=mode)

    assert len(loader) == expected_len
    for _ in range(2):
        # There is no guarantee about the order in which
        # the full vs partial batches will be returned
        # so we need to keep track of which ones we've seen so far
        # and assert at the end.
        full_iterations_left = num_full_iterations_expected
        left_overs_left = left_over_batches.copy()

        for n, d in enumerate(loader):
            print("Dequeued tensor shape ", d.shape)
            if d.shape[0] == batch_size:
                full_iterations_left -= 1
            else:
                assert d.shape[0] in left_overs_left
                left_overs_left.remove(d.shape[0])

        num_iterations_expected = num_full_iterations_expected + len(
            left_over_batches)
        assert full_iterations_left == 0
        assert not left_overs_left
        assert n + 1 == num_iterations_expected


@pytest.mark.parametrize("DatasetType",
                         [IncrementDataset, IncrementIterableDatasetWithLen])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("rebatched_worker_size", [1, 2, 3, 4, None])
def test_rebatched_worker_size(DatasetType, dtype, drop_last,
                               rebatched_worker_size):
    shape = [2, 3]
    num_tensors = 101
    num_workers = 7
    batch_size = 4
    ds = DatasetType(shape, num_tensors, dtype=dtype)
    worker_init_fn = None
    if isinstance(ds, torch.utils.data.IterableDataset):
        worker_init_fn = _worker_init_fn

    if drop_last:
        # Best case expected: floor(101/4) = 25 -> unused = 1
        num_expected = math.floor(num_tensors / batch_size) * batch_size
    else:
        num_expected = num_tensors

    opts = poptorch.Options()
    loader = poptorch.DataLoader(opts,
                                 ds,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 mode=poptorch.DataLoaderMode.AsyncRebatched,
                                 drop_last=drop_last,
                                 rebatched_worker_size=rebatched_worker_size,
                                 worker_init_fn=worker_init_fn)

    values = set()
    for t in loader:
        assert not drop_last or t.shape == torch.Size([4, 2, 3])
        for b in t:
            v = int(b[0][0])
            assert v not in values
            values.add(v)

    assert len(values) == num_expected
    print("Missing tensors:")
    for i in range(num_tensors):
        if i not in values:
            print(i)

    # Make sure it works for more than 1 epoch
    values = set()
    for t in loader:
        assert not drop_last or t.shape == torch.Size([4, 2, 3])
        for b in t:
            v = int(b[0][0])
            assert v not in values
            values.add(v)

    assert len(values) == num_expected


def process_to_kill_asyncdataloader(iterate_over_data: bool):
    """A function executed as a script meant to be killed
    ``test_KeyboardInterrupt_in_async_data_accessor``
    Creates a dataloader and iterates over it.
    """
    # pylint: disable=import-outside-toplevel
    # pylint: disable=reimported
    import time
    import poptorch
    import torch

    opts = poptorch.Options()
    opts.deviceIterations(2)
    opts.replicationFactor(1)
    features = torch.randn([100, 1, 128, 128])
    labels = torch.empty([100], dtype=torch.long).random_(10)
    dataset = torch.utils.data.TensorDataset(features, labels)
    training_data = poptorch.DataLoader(
        opts,
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        mode=poptorch.DataLoaderMode.Async,
    )
    # Empty iteration through the data alters the state of the accessor
    if iterate_over_data:
        for _, _ in training_data:
            pass
    # Needed as a cooldown after the iteration, otherwise the accessor
    # may be in an unsafe state, this is representative of interractive
    # environments.
    time.sleep(1)
    print("[control] Dataloader prepared, waiting for sigint.")

    # Expect the parent process to be force closed in the next 30 seconds
    try:
        time.sleep(30)
        raise RuntimeError(
            "We should not reach this point, we should receive SIGINT before")
    except KeyboardInterrupt:
        print("[control] KeyboardInterrupt received in parent exiting.")


@pytest.mark.parametrize("iterate_over_data", [True, False])
def test_KeyboardInterrupt_in_async_data_accessor(iterate_over_data: bool):
    """ Reproduces an error seen in Jupyter notebooks where dataloader
    Asynchronous Accessors get closed before their controller. Leading
    to error messages being spawned to the notebook command line.

    :args: iterate_over_data: Argument passed to
        ``process_to_kill_asyncdataloader``. Indicates whether to iterate over
        the data or not.
    """
    print("Starting subprocess")
    parent = subprocess.Popen(
        [
            sys.executable,
            "-u",  # needed to ensure messages are sent to stdout immediately
            "-c",
            f"""
import os
# Needed to capture the PID of the AsynchronousDataAccessor
os.environ["POPTORCH_LOG_LEVEL"] = "DEBUG"
import marshal, types
code = marshal.loads({marshal.dumps(process_to_kill_asyncdataloader.__code__)})
fn = types.FunctionType(code, globals(), "kill_this_process")
fn({iterate_over_data})
            """,
        ],
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print("Subprocess started - waiting for signal")

    lines = []
    worker_pid = None
    kill_worker = False
    # Capture the PID of AsynchronousDataAccessor and wait for the signal
    # that the dataloader is ready.
    for line in parent.stdout:
        lines.append(line)
        print("Child - {}".format(line.strip("\n")))
        find_pid = re.match(
            r".*AsynchronousDataAccessor worker process: (\d+)", line)
        if find_pid:
            worker_pid = int(find_pid.group(1))
        if re.match(r"\[control\] Dataloader prepared, waiting for sigint\.",
                    line):
            kill_worker = True
            break

    # Check that both the PID and the signal were caught
    if not kill_worker:
        parent.send_signal(signal.SIGINT)
        raise RuntimeError("The termination signal for the worker process " +
                           "was not received.")
    if worker_pid is None:
        parent.send_signal(signal.SIGINT)
        raise RuntimeError(
            "Could not kill the AsynchronousDataAccessor, its " +
            "PID could not be captured from the standard output.")

    print("Sending SIGINT to ", worker_pid)
    os.kill(worker_pid, signal.SIGINT)
    parent.send_signal(signal.SIGINT)

    for line in parent.stdout:
        lines.append(line)
        print("Child - {}".format(line.strip("\n")))

    unexpected_lines = [
        line for line in lines
        if "[debug]" not in line and "[control]" not in line
    ]
    assert not unexpected_lines, "Unexpected lines in output:\n%s" % "".join(
        unexpected_lines)
