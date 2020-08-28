# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as multiprocessing

import poptorch.poptorch_core as poptorch_core
from poptorch.poptorch_core import ipuHardwareIsAvailable

from .logging import logger
from . import _impl
from .enums import *
from .ops import *
from .options import *


class IPU(nn.Module):
    def __init__(self, ipu_id, layer_to_call=None):
        super().__init__()

        self.ipu_id = ipu_id
        self.layer_to_call = layer_to_call

    def __enter__(self):
        begin_ipu_block(self.ipu_id)

    def __exit__(self, type, value, traceback):
        end_ipu_block()

    def __call__(self, *input, **kwargs):
        begin_ipu_block(self.ipu_id)
        out = self.layer_to_call(*input, **kwargs)
        return out


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 options,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 num_workers=0,
                 **kwargs):
        assert isinstance(options, Options)
        self._combined_batch_size = batch_size * \
            options.device_iterations * \
            options.replication_factor * \
            options.Training.gradient_accumulation

        super().__init__(dataset,
                         batch_size=self._combined_batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         **kwargs)

    @property
    def combinedBatchSize(self):
        return self._combined_batch_size


# A dataloader which launches the dataloading process on a separate thread to
# allow for the data to be preprocessed asynchronous on CPU to minimize CPU/IPU
# transfer time.
class AsynchronousDataAccessor:
    def __init__(self,
                 dataset,
                 buffer_size=3,
                 miss_sleep_time_in_ms=0.1,
                 load_indefinitely=False):
        self._training_data = dataset
        self._buffer_size = buffer_size
        self._miss_sleep_time_in_ms = miss_sleep_time_in_ms
        self._load_indefinitely = load_indefinitely

        # We return shared memory to the user so we can't tell the worker to
        # refill it until the next item is requested.
        self._previously_ready_element = None

        self._data_fetcher = None
        self._ready_to_read_index = None
        self._is_single_tensor = None
        self._data_buffers = None
        self._ring_read_index = 0

    def terminate(self):
        self._data_fetcher.terminate()

    def fetch_data(self, queue):
        dataset_iterator = iter(self._training_data)

        data = next(dataset_iterator)

        # We support either a single tensor or a flat 1D iterable of tensors.
        is_single_tensor = False
        if isinstance(data, torch.Tensor):
            is_single_tensor = True
            data = (data, )

        # We communicate with the host via an array of sentinel values to say
        # if the data is ready as this has much better latency than queue or
        # lock approaches.
        ready_to_read_index = torch.tensor([False] * self._buffer_size,
                                           dtype=torch.bool).share_memory_()
        queue.put(ready_to_read_index)

        data_buffers = []

        # Tell the host how many tensors we will be sending.
        data_length = len(data)

        queue.put(data_length)
        queue.put(is_single_tensor)

        # Send the tensors to the host.
        for index, tensor in enumerate(data):
            assert isinstance(
                tensor,
                torch.Tensor), """Tensor at index %d is not a torch tensor.
                    AsynchronousDataAccessor expects data to
                    be organised as a flat 1D container of
                    tensors.""" % index

            # Shared with parent process.
            memory = tensor.expand(
                self._buffer_size,
                *tensor.size()).clone().contiguous().share_memory_()
            data_buffers.append(memory)

            # Send it to the host.
            queue.put(memory)

        # We've loaded the first element as part of the spin up process.
        ready_to_read_index[0] = True

        ring_write_index = 1

        any_data_sent = True

        while True:
            try:
                # Only pull the next iteration if we sent data the last one,
                # otherwise try send the old one again.
                if any_data_sent:
                    data = next(dataset_iterator)
                    if isinstance(data, torch.Tensor):
                        data = (data, )
            except StopIteration:
                # EOF: Reset the iterator back to the start.
                if self._load_indefinitely:
                    dataset_iterator = iter(self._training_data)
                    continue

                # Else break
                break

            any_data_sent = False

            if not ready_to_read_index[ring_write_index]:
                # Copy the tensor into the preallocated shared memory.
                for index, tensor in enumerate(data):
                    data_buffers[index][ring_write_index].copy_(tensor)

                # Tell the host this data is ready.
                ready_to_read_index[ring_write_index] = True

                # Quit the loop
                any_data_sent = True

                ring_write_index += 1

                # Ring back around.
                if ring_write_index >= self._buffer_size:
                    ring_write_index = 0

            # (Briefly) sleep the thread if we didn't fetch any data.
            if not any_data_sent:
                time.sleep(self._miss_sleep_time_in_ms)

    def __iter__(self):

        # We use a small queue to get the initial data. The latency of
        # deserialising the python data is too high to be used for the
        # actual fetch so we just use this to return the initial buffers
        # in shared memory which will be used for the actual read/write
        # in the hot loop.
        queue = multiprocessing.Queue()

        # Fetch the data on a seperate process.
        self._data_fetcher = multiprocessing.Process(target=self.fetch_data,
                                                     args=(queue, ))
        self._data_fetcher.start()

        self._ready_to_read_index = queue.get(block=True)

        buffer_len = queue.get(block=True)

        self._is_single_tensor = queue.get(block=True)

        self._data_buffers = []

        for _ in range(0, buffer_len):
            # Get the buffer from the host.
            buffer = queue.get(block=True)
            self._data_buffers.append(buffer)

        return self

    def __next__(self):
        data = []

        # Set the previous iteration to false so it can be pulled in now
        # avoiding any data races.
        if self._previously_ready_element is not None:
            self._ready_to_read_index[self._previously_ready_element] = False

        self._previously_ready_element = None
        # We block until the element is ready.
        while self._previously_ready_element is None:

            # Grab the data waiting in the ring buffer.
            if self._ready_to_read_index[self._ring_read_index]:
                # Pull the ready buffer.
                for _, buffer in enumerate(self._data_buffers):
                    data.append(buffer[self._ring_read_index])

                self._previously_ready_element = self._ring_read_index

                self._ring_read_index += 1
                # Ring back around.
                if self._ring_read_index >= self._buffer_size:
                    self._ring_read_index = 0

            # Processed all data and the process is dead, EOF.
            process_dead = not self._data_fetcher.is_alive()
            if self._previously_ready_element is None and process_dead:
                raise StopIteration

        # Return either one tensor or the list.
        if self._is_single_tensor:
            return data[0]

        # Else return the list.
        return data


def trainingModel(model, options=None, optimizer=None):
    return _impl.PoplarExecutor(model=model,
                                options=options,
                                training=True,
                                optimizer=optimizer)


def inferenceModel(model, options=None):
    return _impl.PoplarExecutor(model=model, options=options, training=False)


def propagateInputShapes(graph, dummyInputs):
    for graphInput, dummyInput in zip(graph.inputs(), dummyInputs):
        graphInput.inferTypeFrom(dummyInput)
    poptorch_core.propagateInputShapes(graph)
