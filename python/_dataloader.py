# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import io
import sys
import os
import tempfile
import enum
import math
import pickle
import time
import torch
import torch.multiprocessing as multiprocessing

# Do not import any poptorch.* here: it will break the poptorch module
from . import enums
from ._logging import logger
from . import _impl


class AsynchronousWorker:
    """Interface for the host to create and manage a separate worker process to fetch elements from a dataset."""

    def __init__(self, buffer_size, miss_sleep_time_in_ms, dataset,
                 load_indefinitely, early_preload, sharing_strategy,
                 rebatched_size):
        self._process = _AsynchronousWorkerProcess(
            buffer_size, miss_sleep_time_in_ms, dataset, load_indefinitely,
            early_preload, sharing_strategy, rebatched_size)
        self._was_used = False
        self._worker_started = False

        # Keep end of file events in a special buffer shared between worker and device. This is due to the worker reseting automatically.
        (self._command_pipe, self._is_single_tensor, self._eof,
         self._data_buffers) = self._process.start()

    def terminate(self):
        if self._process.isAlive():
            self._requestShutdown()

        self._process.join()

    def resetIterator(self):
        if self._worker_started and not self._was_used:
            # The current iterator hasn't been used: nothing to do.
            return

        # Reset if:
        # - The EOF was reached and the worker is waiting to know if it
        #   should create a new iterator (load_indefinitely=False)
        # - We're partway through an iteration and we want to restart.
        #
        # Note: there is a race condition where the worker reaches EOF
        # after endOfFile() returned False.
        # The consequence is that reset will be called when it wasn't
        # actually needed. (i.e it won't break anything)

        if self._was_used and (not self.endOfFile() or
                               (self.endOfFile()
                                and not self._process.load_indefinitely)):
            # Request reset:
            self._command_pipe.send(_HostCommand.ResetIterator)
            self.releaseElement()
            # Wait for the worker to acknowledge
            self._eof.waitForReset()
            self._data_buffers.reset()

        self._eof.clearFlag()
        # Let the worker know it can start loading
        self._command_pipe.send(_HostCommand.StartIterating)
        self._was_used = False
        self._worker_started = True

    def dataIsAvailable(self):
        return self._data_buffers.isAvailable()

    def endOfFile(self):
        return self._eof.isEofIndex(self._data_buffers.currentIndex())

    def acquireElementIfAvailable(self):
        assert not self._data_buffers.hasLock(), (
            "The current element "
            "must be released by calling releaseElement() before trying to "
            "acquire a new one")

        # Important: eof must be checked **after** dataIsAvailable.
        #
        # The worker does:
        # 1. setEOFflag()
        # 2. if load_indefinitely -> start prefetching the next iteration.
        # 3. mark data as available.
        #
        # So in the consumer / reader we need to check the flags in reverse
        # order otherwise there is a risk that eof will be False, then by
        # the time data is checked both eof and data are now True but
        # we'll miss eof and iterate over the ring buffer an extra time.
        if not self.dataIsAvailable() or self.endOfFile():
            return None
        left_over = self._eof.leftOver(self._data_buffers.currentIndex())
        # Pull and lock the ready buffer.
        data = self._data_buffers.lock()
        self._was_used = True

        if left_over > 0:
            data = [d.narrow(0, 0, left_over) for d in data]
            # Update the EOF flag to the real index and clear the
            # left over value.
            self._eof.setFlag(self._data_buffers.currentIndex())

        # Return either one tensor or the list.
        if self._is_single_tensor:
            return data[0]

        # Else return the list.
        return data

    def assertNoError(self):
        if not self._process.isAlive():
            assert self._process.exitCode() == 0, \
                "An error occurred in the data fetcher"

    def releaseElement(self):
        # Set the previous iteration to false so it can be pulled in now
        # avoiding any data races.
        self._data_buffers.unlockIfLocked()

    def _requestShutdown(self):
        # Send the exit signal if the worker is still alive.
        try:
            self._command_pipe.send(_HostCommand.Shutdown)
        except BrokenPipeError:
            pass


class _AsynchronousWorkerProcess:
    """Worker process fetching elements from a given dataset"""

    def __init__(self, buffer_size, miss_sleep_time_in_ms, dataset,
                 load_indefinitely, early_preload, sharing_strategy,
                 rebatched_size):
        self._buffer_size = buffer_size
        self._miss_sleep_time_in_ms = miss_sleep_time_in_ms
        self._dataset = dataset
        self.load_indefinitely = load_indefinitely
        self._early_preload = early_preload
        self._process = None
        self._sharing_strategy = sharing_strategy
        self._rebatched_size = rebatched_size
        self._next_batch_idx = 0

    def isAlive(self):
        return self._process.exitcode is None

    def exitCode(self):
        return self._process.exitcode

    def join(self):
        self._process.join(timeout=5)
        # In case it didn't exit cleanly: terminate() it
        self._process.terminate()
        self._process.join()

    def start(self):
        # The dataset might not fit in shared memory: so use the file system instead.
        if self._sharing_strategy != enums.SharingStrategy.FileSystem:
            return self._start()

        # Serialise the dataset to file and replace the dataset by the filename.
        with tempfile.TemporaryDirectory() as d:
            pickle_file = os.path.join(d, "dataset.pkl")
            logger.debug("Serialising dataset to file: %s", pickle_file)
            dataset = self._dataset
            with open(pickle_file, "wb") as f:
                pickle.dump(self._dataset, f)
                self._dataset = pickle_file
            try:
                return self._start()
            finally:
                self._dataset = dataset

    def _start(self):
        assert self._process is None, "Worker already started"
        # We use a small pipe to get the initial data. The latency of
        # deserialising the python data is too high to be used for the
        # actual fetch so we just use this to return the initial buffers
        # in shared memory which will be used for the actual read/write
        # in the hot loop.
        ctx = multiprocessing.get_context('spawn')
        read_data_pipe, write_data_pipe = ctx.Pipe(duplex=False)

        # If the worker exits before the parent process is done
        # setting up the _data_buffers then the pipe will get freed
        # and bad things will happen.
        read_command_pipe, write_command_pipe = ctx.Pipe(duplex=False)

        # Fetch the data on a seperate process.
        logger.debug("AsynchronousDataAccessor parent process: %d",
                     os.getpid())

        self._process = ctx.Process(target=self._mainLoop,
                                    args=(write_data_pipe, read_command_pipe))
        self._process.start()
        write_data_pipe.close()
        read_command_pipe.close()

        try:
            indices_mem = read_data_pipe.recv()
            data_len = read_data_pipe.recv()
            is_single_tensor = read_data_pipe.recv()
            eof_mem = read_data_pipe.recv()
            buffers = _DataRingBufferReader(self._buffer_size, data_len,
                                            indices_mem)

            for data_idx in range(0, data_len):
                # Get the buffer from the host.
                buffer = read_data_pipe.recv()
                buffers.setBuffer(buffer, data_idx)

            # We're all set: let the worker know.
            write_command_pipe.send(_HostCommand.SetupComplete)
            return (write_command_pipe, is_single_tensor,
                    _EndOfFileFlag(eof_mem), buffers)
        except EOFError:
            pass
        # Exit the except block before raising a cleaner exception otherwise the previous one will not be cleared.
        raise _impl.createPoptorchError(
            "AsynchronousDataAccessor worker thread failed to start "
            "(Check above for details)")

    def _mainLoop(self, conn, command_pipe):
        """Main event loop of the asynchronous worker process

        SIGINT signals appear as KeyboardInterrupts and need to be handled
        as the ``atexit`` terminate hook is not guaranteed to be called before
        the signal is propagated to the worker processes.

        See Also:
            :meth:`_mainLoopNoInterrupt` for the implementation of worker event
            loop.
        """
        try:
            return self._mainLoopNoInterrupt(conn, command_pipe)
        except KeyboardInterrupt:
            # Core interpretter libraries may already be unloaded
            # so don't do anything. More detail of caveats in the
            # pytorch note on [ Data Loader Multiprocessing Shutdown Logic ]:
            # https://github.com/pytorch/pytorch/blob/
            # aa7da7b09c4a3f972ede5fd8ad0cbc8c13498a00/
            # torch/utils/data/dataloader.py#L570
            pass

    def _mainLoopNoInterrupt(self, conn, command_pipe):  # pylint: disable=too-many-statements
        # Make sure this process's output gets printed (In case of error)
        sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0),
                                      write_through=True)
        sys.stderr = io.TextIOWrapper(open(sys.stderr.fileno(), 'wb', 0),
                                      write_through=True)

        # We're in a new process: we need to re-initialise the logger
        from ._logging import logger  # pylint: disable=import-outside-toplevel
        logger.debug("AsynchronousDataAccessor worker process: %d",
                     os.getpid())
        # If the dataset is a string then it's a path to file containing
        # the dataset
        if isinstance(self._dataset, str):
            with open(self._dataset, "rb") as f:
                self._dataset = pickle.load(f)
        dataset_iterator = iter(self._dataset)
        rebatched_drop_last = getattr(self._dataset, "rebatched_drop_last",
                                      True)

        data = None
        try:
            data = next(dataset_iterator)
        except StopIteration:
            pass
        if data is None:
            raise _impl.createPoptorchError("The Dataset is empty")

        # We support either a single tensor or a flat 1D iterable of tensors.
        is_single_tensor = False
        if isinstance(data, torch.Tensor):
            is_single_tensor = True
            data = (data, )

        # Tell the host how many tensors we will be sending.
        data_length = len(data)

        buffers = _DataRingBufferWriter(self._buffer_size, data_length)
        # We communicate with the host via an array of sentinel values to say
        # if the data is ready as this has much better latency than queue or
        # lock approaches.
        conn.send(buffers.indices_mem)

        conn.send(data_length)
        conn.send(is_single_tensor)

        eof = _EndOfFileFlag(eof_mem=None)
        conn.send(eof.eof_mem)

        # Send the tensors to the host.
        for index, tensor in enumerate(data):
            assert isinstance(
                tensor,
                torch.Tensor), ("Tensor at index %d is not a torch tensor."
                                " AsynchronousDataAccessor expects data to "
                                "be organised as a flat 1D container of "
                                "tensors.") % index

            # Shared with parent process.
            tensor_size = [*tensor.size()]
            if self._rebatched_size:
                self._next_batch_idx = tensor_size[0]
                # Reshape with repeat if expand is not working in batch dimension
                if tensor_size[0] != self._rebatched_size:
                    repeat_count = math.ceil(self._rebatched_size /
                                             tensor_size[0])
                    # Repeat then shrink to the right size
                    tensor = tensor.repeat(
                        repeat_count,
                        *[1] * (len(tensor_size) - 1))[:self._rebatched_size]
                    tensor_size[0] = self._rebatched_size
            memory = tensor.expand(
                self._buffer_size,
                *tensor_size).clone().contiguous().share_memory_()

            buffers.setBuffer(memory, index)
            # Send it to the host.
            conn.send(memory)

        # We've loaded the first element as part of the spin up process.
        if self._rebatched_size is None or \
                self._next_batch_idx == self._rebatched_size:
            self._next_batch_idx = 0
            # Tell the host this data is ready.
            buffers.markWriteComplete()

        host_handler = _HostCommandHandler(command_pipe)

        if self._early_preload:
            state = _WorkerState.Prefetching
        else:
            state = _WorkerState.Stopped

        rebatch_leftover = []
        while not host_handler.shutdown_now:
            # Check for messages from the parent process:
            host_handler.checkMessages()
            if state == _WorkerState.Stopped:
                if host_handler.waitUntilStartIteration():
                    state = _WorkerState.Loading
                # else reset or shutdown received: fallthrough
            elif state == _WorkerState.Prefetching and \
                    host_handler.startIteratingPending():
                # The host sent a request to start loading so transition from prefetching
                # to loading.
                state = _WorkerState.Loading
            if host_handler.shutdown_now:
                continue
            if host_handler.resetIteratorPending():
                logger.debug("AsynchronousDataAccessor worker: reset command "
                             "received. Creating a new iterator")
                buffers.reset()
                dataset_iterator = iter(self._dataset)
                self._next_batch_idx = 0
                rebatch_leftover = []

                # Let the host know everything has been reset
                eof.setResetFlag()

                # Wait for the host to ask for the new iteration to start
                if not host_handler.waitUntilStartIteration():
                    continue  # received a shutdown command

                logger.debug("AsynchronousDataAccessor worker: the iterator "
                             "has been reset")
                state = _WorkerState.Loading

            # We're now guaranteed to be either loading or prefetching
            eof_reached = False
            # Handle the left overs if any before asking for more data.
            if rebatch_leftover:
                data = rebatch_leftover
                rebatch_leftover = []
            else:
                try:
                    # Retrieve data from the dataset
                    data = next(dataset_iterator)
                    if isinstance(data, torch.Tensor):
                        data = (data, )
                except StopIteration:
                    logger.debug(
                        "AsynchronousDataAccessor worker: end of dataset"
                        " reached")
                    eof_reached = True

            # Wait for a writing slot to become available
            while not buffers.isAvailable(
            ) and not host_handler.priorityCommandWaiting():
                # (Briefly) sleep the thread if we neither is True.
                if self._miss_sleep_time_in_ms > 0.0:
                    time.sleep(self._miss_sleep_time_in_ms)
                host_handler.checkMessages()
            if host_handler.priorityCommandWaiting():
                continue

            if eof_reached:
                # Note: it's important to have a writing slot before signaling
                # the end of the dataset or we might encounter the case where
                # the whole ring buffer is ready to read:
                # [ True, True, True]
                # At that point the read and write indices point at the same
                # index so if we set the EOF as the current write index then
                # the consumer will discard the whole ring buffer instead of
                # consuming the ready to read elements first.
                # Having a writing slot available ensures the read and write
                # indices never match (Even though the slot might not be used).

                # If we reach the EOF before the host asked us to start loading,
                # wait here to avoid potentially overwriting a pending
                # EOF event.
                if state == _WorkerState.Prefetching:
                    if not host_handler.waitUntilStartIteration():
                        continue  # reset or shutdown

                if self._rebatched_size and not rebatched_drop_last \
                    and self._next_batch_idx != 0:
                    eof.setFlag(buffers.currentIndex(), self._next_batch_idx)
                    # We're in the middle of a rebatch so the buffer
                    # should already be available from previous
                    # batch indices.
                    assert buffers.isAvailable()
                    buffers.markWriteComplete()
                else:
                    eof.setFlag(buffers.currentIndex(), 0)

                # If we are not to load indefinitely we wait for the host
                # to explicitly ask for a new iterator to be created.
                if not self.load_indefinitely:
                    logger.debug(
                        "AsynchronousDataAccessor worker: end of dataset"
                        " reached signaled to host: waiting for command from"
                        " host")
                    state = _WorkerState.Stopped
                    continue  # Go back to the wait for reset

                logger.debug("AsynchronousDataAccessor worker: end of dataset "
                             "reached. Creating a new iterator")
                state = _WorkerState.Prefetching

                # We reset and keep the worker thread prefetching.
                dataset_iterator = iter(self._dataset)
                self._next_batch_idx = 0

                logger.debug(
                    "AsynchronousDataAccessor worker: new iterator ready")
                continue

            # We've got a writing slot
            if self._rebatched_size:
                assert not rebatch_leftover, (
                    "Rebatch data should be empty and"
                    " ready to be used if needed")
                for index, tensor in enumerate(data):
                    # Note _index_copy_ doesn't work for FP16, it causes
                    # the following error:
                    # RuntimeError: _th_index_copy_ not supported on CPUType
                    # for Half"
                    #
                    # That's why we instead use a regular copy_
                    in_size = len(tensor)
                    out_size = self._rebatched_size - self._next_batch_idx
                    copy_size = min(in_size, out_size)
                    if in_size > out_size:
                        rebatch_leftover.append(tensor[copy_size:])

                    buffers.current[index][self._next_batch_idx:self.
                                           _next_batch_idx + copy_size].copy_(
                                               tensor[:copy_size])

                self._next_batch_idx += copy_size
            else:
                # Copy the tensor into the preallocated shared memory.
                for index, tensor in enumerate(data):
                    buffers.current[index].copy_(tensor)

            # If we're not rebatching: always notify the host an element is ready.
            # Otherwise only notify the host if the full batch is ready.
            if self._rebatched_size is None or \
                    self._next_batch_idx == self._rebatched_size:
                self._next_batch_idx = 0
                # Tell the host this data is ready.
                buffers.markWriteComplete()

        logger.debug(
            "AsynchronousDataAccessor worker: ready to exit: checking parent"
            " is ready")
        # In the unlikely event the worker is done reading the dataset
        # before the parent is done setting the buffers up: wait here.
        host_handler.waitUntilSetupComplete()
        logger.debug("AsynchronousDataAccessor worker: clean exit")


class _HostCommand(enum.IntEnum):
    SetupComplete = 0
    Shutdown = 1
    ResetIterator = 2
    StartIterating = 3


class _WorkerState(enum.IntEnum):
    Stopped = 0
    Prefetching = 1
    Loading = 2


class _HostCommandHandler:
    def __init__(self, command_pipe):
        self.pipe = command_pipe
        self.setup_complete = False
        self.shutdown_now = False
        self._reset_iterator = False
        self._start_iterating = False

    def checkMessages(self, blocking=False, ignore_setup_complete=True):
        """
        ignore_setup_complete: setup complete is usually just noise. (We only
        care about the setup being complete if we're trying trying to shutdown
        the worker process), so when asked to wait for a message, if the first
        one we receive is setup complete, usually we'll want to wait some more
        for the one we actually are interested in.
        """
        # Check for messages from the parent process:
        if self.pipe.poll() or blocking:
            cmd = self.pipe.recv()  # remove the data
            assert isinstance(cmd, _HostCommand)
            if cmd == _HostCommand.SetupComplete:
                logger.debug("SetupComplete command received")
                assert not self.setup_complete, ("More than one SetupComplete "
                                                 "event received")
                self.setup_complete = True
                if ignore_setup_complete:
                    self.checkMessages(blocking)
            elif cmd == _HostCommand.Shutdown:
                logger.debug("Shutdown command received")
                self.shutdown_now = True
            elif cmd == _HostCommand.ResetIterator:
                logger.debug("ResetIterator command received")
                self._reset_iterator = True
            elif cmd == _HostCommand.StartIterating:
                logger.debug("StartIterating command received")
                self._start_iterating = True
            else:
                raise _impl.createPoptorchError(
                    f"Unknown command received {cmd}")

    def priorityCommandWaiting(self):
        return self.shutdown_now or self._reset_iterator

    def waitUntilSetupComplete(self):
        if not self.setup_complete:
            self.checkMessages(blocking=True, ignore_setup_complete=False)
        # Shutdown has been requested: there is no other valid command the host
        # can send at that point
        assert self.setup_complete

    def startIteratingPending(self):
        """Note: returns state and reset the value to False"""
        if self._start_iterating:
            self._start_iterating = False
            return True
        return False

    def resetIteratorPending(self):
        """Note: returns state and reset the value to False"""
        if self._reset_iterator:
            self._reset_iterator = False
            return True
        return False

    def waitUntilStartIteration(self):
        """Wait until a start iteration message is received.

        Return True if we successfully received a start iteration message.
        False if it was a reset or shutdown command.
        """
        if self.priorityCommandWaiting():
            return False
        if not self._start_iterating:
            self.checkMessages(blocking=True)

        return self.startIteratingPending()


class _EndOfFileFlag:
    """
    Share a small 2 values buffer with host to signal EOF and where in ring
    buffer the event occurred.

    First value:
    -1 means no event and the worker will keep loading until EOF is
    reached or the buffer is full.

    -2 means iterator reset complete. (Will be cleared by the worker once it's
    received the start iterating command from the host)

    Any other value: wait for an iterator to be created to start
    loading more data.

    Second value: when rebatching + drop_last=False:
    Indicate the batch size of the left over tensor

    0: No left over
    0 < N < rebatch_size: left-over batch size

    """

    def __init__(self, eof_mem=None):
        if eof_mem is None:
            eof_mem = torch.tensor([-1, 0], dtype=torch.int).share_memory_()
        self.eof_mem = eof_mem

    def setResetFlag(self):
        """Called by the worker once the iterator has been reset"""
        self.eof_mem[0] = -2

    def waitForReset(self):
        while self.eof_mem[0] != -2:
            pass

    def isEofIndex(self, index):
        return self.eof_mem[0] == index and self.eof_mem[1] == 0

    def leftOver(self, index):
        """Batch size of the tensor at the end of file index.

        0 either means it's not the end of the dataset yet or
        there is no left over batches, the end of file index is empty.
        (It will contain the first element from the next iteration if a new
        iterator is created).

        N means the element at the end of file index has a reduced batch of N.
        (The first element from the next iteration if a new iterator is
        created will be located at the next index).
        """
        if self.eof_mem[0] == index:
            return self.eof_mem[1]
        return 0

    def clearFlag(self):
        self.eof_mem[1] = 0
        self.eof_mem[0] = -1

    def setFlag(self, buffer_idx, last_batch_size=0):
        """If ``last_batch_size`` is 0 then ``buffer_idx`` is the index of the
        first buffer after the end of file.

        Otherwise the buffer at ``buffer_idx`` will contain a tensor of reduced
        batch size ``last_batch_size`` elements. (Only used when drop_last=False
        and rebatched_size > 0).
        """
        # Important: eof_tensor[1] must be set before eof_tensor[0]
        # to avoid race conditions with the consumer.
        self.eof_mem[1] = last_batch_size
        self.eof_mem[0] = buffer_idx


class _RingBufferIndex:
    """The index ring buffer is a ``buffer_size`` list of booleans keeping track
    which elements from the data ring buffers is ready to be written or ready to
    be read.

    * True: ready to write
    * False: ready to read.

    It is allocated using shared memory as it is shared between the worker
    process (producer) and the main process (consumer).

    The memory for the ring buffer will be allocated by the producer (Worker
    process) and initialised to all False (i.e all ready to be written).
    """

    def __init__(self, buffer_size, indices_mem=None):
        if indices_mem is not None:
            self.buffers = indices_mem
            assert len(indices_mem) == buffer_size
        else:
            self.buffers = torch.tensor([False] * buffer_size,
                                        dtype=torch.bool).share_memory_()
        self.buffer_size = buffer_size
        self._index = 0

    def increment(self):
        self._index += 1
        if self._index >= self.buffer_size:
            self._index = 0

    def reset(self):
        self.buffers.fill_(False)
        self._index = 0

    def set(self, value):
        self.buffers[self._index] = value

    def value(self):
        return self.buffers[self._index]

    def __call__(self):
        return self._index


class _IDataRingBuffer:
    def __init__(self, buffer_size, data_len, indices_mem=None):
        self._index = _RingBufferIndex(buffer_size, indices_mem)
        D = data_len
        B = buffer_size
        assert buffer_size == self._index.buffer_size
        # The structure of the allocated buffers is
        # buffers[D][B][tensor] where:
        # D = number of tensors in one tuple from the dataset
        # B = number of buffers in ring buffer.
        #
        # but we're going to iterate over B so we will store
        # the buffers as they get added to:
        # buffers[B][D][tensor]
        self._data = [[None] * D for _ in range(B)]

    def setBuffer(self, buffer, data_idx):
        """Add a new buffer to the ring
        expecting the tensor to be of the shape
        buffer[B][tensor] but we store tensors as:
        buffers[B][D] so we need to shuffle the data.
        """
        assert len(buffer) == self._index.buffer_size
        assert data_idx < len(self._data[0])
        for d in range(self._index.buffer_size):
            self._data[d][data_idx] = buffer[d]

    @property
    def current(self):
        """Return the current buffer"""
        return self._data[self._index()]

    @property
    def indices_mem(self):
        """Return the shared memory buffer used
        to store the indices"""
        return self._index.buffers

    def currentIndex(self):
        return self._index()

    def reset(self):
        """Reset the state of the ring buffer
        (All the buffers become available to write again)"""
        self._index.reset()


class _DataRingBufferWriter(_IDataRingBuffer):
    """The writer's logic goes as follow:

        - Wait for the current slot to become available for writing
        - Fill the buffer
        - Mark the buffer as ready to be read and move to the next one.

        >>> while True:
        ...     while not buffers.isAvailable():
        ...         time.sleep()
        ...     buffers.current.copy(data)
        ...     buffers.markWriteComplete()
    """

    def markWriteComplete(self):
        """Mark the current buffer as ready to
        be read and move to the next buffer."""
        self._index.set(True)
        self._index.increment()

    def isAvailable(self):
        """Return True if the current index is available for writing,
        or False if it contains a tensor which hasn't been read by the
        consumer process yet."""
        return not bool(self._index.value())


class _DataRingBufferReader(_IDataRingBuffer):
    """The reader's logic goes as follow:

        - Wait for the current slot to become ready to read.
        - Mark the buffer as locked for reading and move to next buffer.
        - Read the locked buffer
        - Release the locked buffer

        Note: the consumer can check if the current buffer is available
        while the previous one is still locked however it cannot lock
        more than one buffer at any given time.
    """

    def __init__(self, buffer_size, data_len, indices_mem=None):
        self._locked = None
        super().__init__(buffer_size, data_len, indices_mem)

    def isAvailable(self):
        """Return True if the current buffer is ready to be read."""
        return bool(self._index.value())

    def hasLock(self):
        """Return True if the ring buffer currently has a buffer
        locked for reading."""
        return self._locked is not None

    def lock(self):
        assert self._locked is None
        self._locked = self.currentIndex()
        data = self.current
        self._index.increment()
        return data

    def unlockIfLocked(self):
        if self._locked is not None:
            self._index.buffers[self._locked] = False
        self._locked = None
