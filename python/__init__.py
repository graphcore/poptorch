# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import atexit
import io
import os
import sys
import time

import torch
import torch.nn as nn
import torch.multiprocessing as multiprocessing

from .logging import logger

assert torch.__version__.startswith("@TORCH_VERSION@"), (
    "This version"
    " of PopTorch only works with torch==@TORCH_VERSION@ but the version "
    f"installed is {torch.__version__}")
import poptorch.poptorch_core as poptorch_core
from poptorch.poptorch_core import ipuHardwareIsAvailable, setLogLevel

from . import _impl
from .enums import *
from .ops import *
from .options import *
from . import distributed
from ._impl import PoplarExecutor
from . import optim

__version__ = "@VERSION@-@SNAPSHOT@"


class _RepeatSampler:
    """ Sampler that repeats forever.

    Args:
        real_sampler (Sampler)
    """

    def __init__(self, real_sampler, is_iterable):
        self.real_sampler = real_sampler
        self.is_iterable = is_iterable

    def __iter__(self):
        while True:
            yield from iter(self.real_sampler)
            if self.is_iterable:
                yield self  # Indicate the end of the dataset

    def __len__(self):
        return len(self.real_sampler)


class DataLoader(torch.utils.data.DataLoader):
    """ Thin wrapper around the traditional `torch.utils.data.DataLoader` to
    abstract away some of the batch sizes calculations.

    If this DataLoader is used in a distributed execution environment, it will
    ensure that each process uses a different subset of the dataset.
    """

    def __init__(self,
                 options,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 num_workers=0,
                 drop_last=True,
                 persistent_workers=None,
                 **kwargs):
        """
        :param poptorch.Options options: Options that will be used to compile
            and run the model.
        :param dataset: The dataset to get the data from.
        :param int batch_size: This is the batch size in the conventional sense
            of being the size that runs through an operation in the model at
            any given time.
        :param bool shuffle: Whether or not the dataset should be shuffled.
        :param int num_workers: Number of worker processes to use to read the
            data.
        :param bool drop_last: If True and the number of elements in the
            dataset is not a multiple of the combined batch size then the
            incomplete batch at the end will be dropped.
        :param bool persistent_workers: Re-use workers between
            iterations if True.
            If None (default): enabled if num_workers > 0, disabled otherwise.
        :param kwargs: Other options to pass to the Torch's DataLoader's
            constructor.
        """
        assert isinstance(options, Options)
        if persistent_workers is None:
            persistent_workers = num_workers > 0

        if batch_size is None:
            self._combined_batch_size = None
        else:
            self._combined_batch_size = batch_size * \
                options.device_iterations * \
                options.replication_factor * \
                options.Training.gradient_accumulation
            self._options = options

        # Iterable datasets need to be handled differently: they don't have __getitem__ and __len__
        self._is_iterable = isinstance(dataset,
                                       torch.utils.data.IterableDataset)

        if self._is_iterable:
            assert options.Distributed.numProcesses == 1, (
                "IterableDatasets "
                "not supported for distributed execution")
        else:
            num_elts = len(dataset)
            assert drop_last or self._combined_batch_size is None or \
                num_elts % (self._combined_batch_size *
                            options.Distributed.numProcesses) == 0, (
                f"The number of elements in the dataset ({num_elts}) is not "
                "divisible by the number of elements processed per step "
                f'''({self._combined_batch_size *
                        options.Distributed.numProcesses})'''
                " and drop_last=False. Switch to drop_last=True.")

            if options.Distributed.numProcesses > 1:
                assert not shuffle or options.exists("random_seed"), (
                    "When using distributed execution you must set "
                    "poptorch.Options.randomSeed()")
                assert self._combined_batch_size is not None, (
                    "batch_size=None not allowed for distributed"
                    " execution.")

                class _SubDataset:
                    def __init__(self, dataset, opts, step):
                        num_elts = len(dataset)
                        per_proc = step * (
                            num_elts // (step * opts.Distributed.numProcesses))
                        self._offset = opts.Distributed.processId * per_proc
                        self._length = min(per_proc, num_elts - self._offset)
                        self._dataset = dataset

                    def __len__(self):
                        return self._length

                    def __getitem__(self, index):
                        return self._dataset[index + self._offset]

                dataset = _SubDataset(dataset, options,
                                      self._combined_batch_size)

        super().__init__(dataset,
                         batch_size=self._combined_batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         drop_last=drop_last,
                         **kwargs)
        # PyTorch's sampler creates / deletes workers for each iteration through
        # the dataset.
        # In order to reuse the workers between epochs we use our own infinite
        # sampler.
        if persistent_workers:
            if self.batch_sampler is not None:
                object.__setattr__(
                    self, "batch_sampler",
                    _RepeatSampler(self.batch_sampler, self._is_iterable))
            if self.sampler is not None:
                object.__setattr__(
                    self, "sampler",
                    _RepeatSampler(self.sampler, self._is_iterable))
            # The iterator cannot be pickled, so create it in __iter__()
        self._infinite_iterator = None
        self._persistent_workers = persistent_workers

    def __iter__(self):
        if not self._persistent_workers:
            yield from super().__iter__()
            return
        if self._infinite_iterator is None:
            self._infinite_iterator = super().__iter__()

        if self._is_iterable:
            # Return a single epoch long iterator
            while True:
                value = next(self._infinite_iterator)  # pylint: disable=stop-iteration-return
                if value == self.batch_sampler:
                    # The sampler returns itself to indicate the end of the dataset
                    # See class _RepeatSampler
                    break
                yield value
        else:
            for _ in range(len(self)):
                yield next(self._infinite_iterator)  # pylint: disable=stop-iteration-return

    @property
    def combinedBatchSize(self):
        """Total number of elements consumed from the dataset for a single
        execution of the model."""
        return self._combined_batch_size

    @property
    def options(self):
        """A reference to the options that were used to initialise this
        DataLoader.
        """
        return self._options


class AsynchronousDataAccessor:
    """A dataloader which launches the dataloading process on a separate thread
    to allow for the data to be preprocessed asynchronous on CPU to minimize
    CPU/IPU transfer time.

    This works by loading the data into a ring buffer of shared memory.
    When the IPU needs another batch it uses the data ready in the in
    the ring buffer. The memory is shared so will be used inplace and
    won't be freed until the next batch is requested. Behind the scenes
    the worker thread will be filling the unready elements of the ring
    buffer.

    .. important:: In order to avoid hanging issues related to ``OpenMP`` and
        ``fork()`` the ``AsynchronousDataAccessor`` uses the ``spawn`` start
        method which means your dataset must be serializable by ``pickle``.
        For more information see
        https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    """

    def __init__(self,
                 dataset,
                 buffer_size=3,
                 miss_sleep_time_in_ms=0.1,
                 load_indefinitely=True):
        """
        :param dataset: The dataset to pull data from, this can be any Python
            iterable.
        :param buffer_size: The size of the ring buffer.
        :param miss_sleep_time_in_ms: When the buffer is full how long should
            we sleep the worker before checking again.
        :param load_indefinitely: If True when we hit the end of the dataset
            we will just loop round again.
        """

        # To avoid hangs when the application exits: implicitly call terminate().
        atexit.register(self.terminate)
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

        # Keep end of file events in a special buffer shared between worker and device. This is due to the worker reseting automatically.
        self._eof_event_tensor = None

        self._ring_read_index = 0

        # Start the worker process.
        self.spin_up()

    def terminate(self):
        """
        An override function to kill the worker process manually.
        """
        if self._data_fetcher:
            if self._data_fetcher.exitcode is None:
                # Send the exit signal if the worker is still alive.
                try:
                    self._pipe.send(0)
                except BrokenPipeError:
                    pass
            self._data_fetcher.join(timeout=5)
            # In case it didn't exit cleanly: terminate() it
            self._data_fetcher.terminate()
            self._data_fetcher.join()
            self._data_fetcher = None

    def __del__(self):
        self.terminate()

    def __len__(self):
        return len(self._training_data)

    def fetch_data(self, conn, pipe):  # pylint: disable=too-many-statements
        # Make sure this process's output gets printed (In case of error)
        sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0),
                                      write_through=True)
        sys.stderr = io.TextIOWrapper(open(sys.stderr.fileno(), 'wb', 0),
                                      write_through=True)
        shutdown_now = False
        setup_complete = False

        logger.debug("AsynchronousDataAccessor worker process: %d",
                     os.getpid())
        dataset_iterator = iter(self._training_data)

        data = None
        try:
            data = next(dataset_iterator)
        except StopIteration:
            pass
        if data is None:
            raise RuntimeError("The Dataset is empty")

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
        conn.send(ready_to_read_index)

        data_buffers = []

        # Tell the host how many tensors we will be sending.
        data_length = len(data)

        conn.send(data_length)
        conn.send(is_single_tensor)

        # Share a small buffer with host to signal EOF and where in ring buffer the event occured.
        # -1 means no event and the worker will keep loading. We start with a dummy event so the
        # EOF won't be hit before the first call to __init__
        eof_tensor = torch.tensor([42], dtype=torch.int).share_memory_()
        conn.send(eof_tensor)

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
            conn.send(memory)

        # We've loaded the first element as part of the spin up process.
        ready_to_read_index[0] = True

        ring_write_index = 1

        any_data_sent = True

        while not shutdown_now:
            # Check for messages from the parent process:
            if pipe.poll():
                pipe.recv()  # remove the data
                # First message is to indicate the setup is complete
                # the second message is to request shutdown.
                if setup_complete:
                    shutdown_now = True
                    continue
                else:
                    setup_complete = True
            # If we hit EOF sleep till re-awakened by host
            if eof_tensor[0] != -1:
                time.sleep(self._miss_sleep_time_in_ms)
                continue

            try:

                # Only pull the next iteration if we sent data the last one,
                # otherwise try send the old one again.
                if any_data_sent:
                    data = next(dataset_iterator)
                    if isinstance(data, torch.Tensor):
                        data = (data, )
            except StopIteration:
                # If we are not to load indefinitely we just kill the worker.
                if not self._load_indefinitely:
                    logger.debug(
                        "AsynchronousDataAccessor worker: end of dataset"
                        " reached")
                    break

                logger.debug(
                    "AsynchronousDataAccessor worker: end of dataset reached."
                    " Creating a new iterator")
                # We always reset and will keep the worker thread running.
                dataset_iterator = iter(self._training_data)

                # Tell the host where the EOF occured.
                eof_tensor[0] = ring_write_index
                logger.debug(
                    "AsynchronousDataAccessor worker: new iterator ready")
                continue

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

        logger.debug(
            "AsynchronousDataAccessor worker: ready to exit: checking parent"
            " is ready")
        # In the unlikely event the worker is done reading the dataset
        # before the parent is done setting the buffers up: wait here.
        if not setup_complete:
            pipe.recv()
        logger.debug("AsynchronousDataAccessor worker: clean exit")

    def spin_up(self):
        # We have already spun up the worker.
        if self._data_fetcher is not None:
            return
        # We use a small queue to get the initial data. The latency of
        # deserialising the python data is too high to be used for the
        # actual fetch so we just use this to return the initial buffers
        # in shared memory which will be used for the actual read/write
        # in the hot loop.
        ctx = multiprocessing.get_context('spawn')
        queue, child_queue = ctx.Pipe()

        # If the worker exits before the parent process is done
        # setting up the _data_buffers then the queue will get freed
        # and bad things will happen.
        self._pipe, child_setup_complete = ctx.Pipe()

        # Fetch the data on a seperate process.
        logger.debug("AsynchronousDataAccessor parent process: %d",
                     os.getpid())
        self._data_fetcher = ctx.Process(target=self.fetch_data,
                                         args=(child_queue,
                                               child_setup_complete))
        self._data_fetcher.start()
        child_queue.close()
        child_setup_complete.close()
        try:
            self._ready_to_read_index = queue.recv()
            buffer_len = queue.recv()
            self._is_single_tensor = queue.recv()
            self._eof_event_tensor = queue.recv()
            self._data_buffers = []

            for _ in range(0, buffer_len):
                # Get the buffer from the host.
                buffer = queue.recv()
                self._data_buffers.append(buffer)

            # We're all set: let the worker know.
            self._pipe.send(0)
            return
        except EOFError:
            pass
        # Exit the except block before raising a cleaner exception otherwise the previous one will not be cleared.
        raise RuntimeError(
            "AsynchronousDataAccessor worker thread failed to start "
            "(Check above for details)")

    def __iter__(self):
        self._eof_event_tensor[0] = -1
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

            # We didn't fetch any data this time. Check to see if worker is dead or EOF.
            if self._previously_ready_element is None:

                # Processed all data and the process is dead, EOF.
                process_dead = not self._data_fetcher.is_alive()

                if process_dead:
                    assert self._data_fetcher.exitcode == 0, \
                            "An error occurred in the data fetcher"
                    raise StopIteration

                # Check if there has been an EOF event.
                if self._eof_event_tensor[0] != -1:
                    raise StopIteration

        # Return either one tensor or the list.
        if self._is_single_tensor:
            return data[0]

        # Else return the list.
        return data


def trainingModel(model, options=None, optimizer=None):
    """ Create a PopTorch training model, from a PyTorch model, to run on IPU
    hardware in training mode.

    :param torch.nn.Module model: The PyTorch model to wrap.
    :param poptorch.Options options: The IPU specific options
    :param torch.optim.Optimizer optimizer: The optimizers to apply during
           training.
        Supported optimizers: ``optim.SGD``, ``optim.AdamW``, ``optim.RMSprop``,
                              ``optim.LAMB``.
    :returns: The :py:class:`poptorch.PoplarExecutor` wrapper to use in place
        of ``model``.
    """

    # To understand which variable groups the user wants to apply the
    # optimizer to we need to mark them via a wrapper. We do this because
    # when we reference the variables in the context of the operation we
    # get the corresponding IR value for "free" as part of the trace.
    # Otherwise we would need a system to map the variable in the optimizer
    # to the variable in the model to the variable in the IR.
    class OptimizerWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args, **kwargs):
            out = self.model(*args, **kwargs)
            apply_optimizer(optimizer)
            return out

    maybe_wrapped_model = model

    if optimizer and len(optimizer.param_groups) > 1:
        maybe_wrapped_model = OptimizerWrapper(model)

    return PoplarExecutor(model=maybe_wrapped_model,
                          options=options,
                          training=True,
                          optimizer=optimizer,
                          user_model=model)


def inferenceModel(model, options=None):
    """ Create a PopTorch inference model, from a PyTorch model, to run on IPU
    hardware in inference mode.

    :param torch.nn.Module model: The PyTorch model to wrap.
    :param poptorch.Options options: The IPU specific options
    :returns: The :py:class:`poptorch.PoplarExecutor` wrapper to use in place
        of ``model``.
    """
    return PoplarExecutor(model=model, options=options, training=False)


def propagateInputShapes(graph, dummyInputs):
    for graphInput, dummyInput in zip(graph.inputs(), dummyInputs):
        graphInput.inferTypeFrom(dummyInput)
    poptorch_core.propagateInputShapes(graph)
