# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import atexit
import torch

# These are needed before the assert
# pylint: disable=wrong-import-order
from . import _logging
from ._logging import logger
# pylint: enable=wrong-import-order

assert torch.__version__.startswith("@TORCH_VERSION@"), (
    "This version"
    " of PopTorch only works with torch==@TORCH_VERSION@ but the version "
    f"installed is {torch.__version__}")

import poptorch.poptorch_core as poptorch_core

from . import _impl
from .enums import *
from .ops import *
from .options import *
from ._impl import PoplarExecutor
from . import optim
from . import profiling

__version__ = "@VERSION@-@SNAPSHOT@"


class _RepeatSampler(torch.utils.data.IterableDataset):
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
                yield {
                    "StopIteration": True
                }  # Indicate the end of the dataset

    def __len__(self):
        return len(self.real_sampler)


class _SubDataset:
    def __init__(self, dataset, opts, step):
        num_elts = len(dataset)
        per_proc = step * (num_elts // (step * opts.Distributed.numProcesses))
        self._offset = opts.Distributed.processId * per_proc
        self._length = min(per_proc, num_elts - self._offset)
        self._dataset = dataset

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self._dataset[index + self._offset]


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

        # Iterable datasets need to be handled differently: they don't have
        # __getitem__ and __len__
        self._is_iterable = isinstance(dataset,
                                       torch.utils.data.IterableDataset)

        if self._is_iterable:
            assert options.Distributed.numProcesses == 1, (
                "IterableDatasets "
                "not supported for distributed execution")
            # TODO(T30952: Remove assert once persistent_workers is handled
            # by upstream Torch.
            assert num_workers < 2 or not persistent_workers, (
                "Currently IterableDatasets do not support num_workers > 1 "
                "and persistent_workers=True")
            if num_workers > 1 and "worker_init_fn" not in kwargs:
                logger.warning(
                    "IterableDataset used with num_workers="
                    "%d but no worker_init_fn specified: as a result"
                    " the DataLoader will return %d times each element"
                    " in the dataset (See torch.utils.data.IterableDataset's"
                    " documentation for more information)", num_workers,
                    num_workers)
        else:
            num_elts = len(dataset)
            assert drop_last or self._combined_batch_size is None or \
                num_elts % (self._combined_batch_size *
                            options.Distributed.numProcesses) == 0, (
                                f"The number of elements in the dataset "
                                "({num_elts}) is not divisible by the number of"
                                " elements processed per step "
                                f'''({self._combined_batch_size *
                                options.Distributed.numProcesses})'''
                                " and drop_last=False. Switch to "
                                "drop_last=True.")

            if options.Distributed.numProcesses > 1:
                assert not shuffle or options.exists("random_seed"), (
                    "When using distributed execution you must set "
                    "poptorch.Options.randomSeed()")
                assert self._combined_batch_size is not None, (
                    "batch_size=None not allowed for distributed"
                    " execution.")

                dataset = _SubDataset(dataset, options,
                                      self._combined_batch_size)
        # IterableDatasets don't use indices so wrap the dataset in a
        # _RepeatSampler instead.
        if self._is_iterable and persistent_workers:
            dataset = _RepeatSampler(dataset, self._is_iterable)
        if not self._is_iterable:
            dataset = profiling.Channel("dataset").instrument(
                dataset, "__getitem__")

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
        # Note: PyTorch already uses an infinite sampler for iterable datasets
        # so don't touch it.
        if persistent_workers and not self._is_iterable:
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

    @property
    def _profiling(self):
        return profiling.Channel("poptorch.DataLoader")

    def __iter__(self):
        if not self._persistent_workers:
            yield from self._profiling.instrument(super().__iter__(),
                                                  "__next__")
            return
        if self._infinite_iterator is None:
            self._infinite_iterator = self._profiling.instrument(
                super().__iter__(), "__next__")

        if self._is_iterable:
            # Return a single epoch long iterator
            while True:
                value = next(self._infinite_iterator)  # pylint: disable=stop-iteration-return
                if isinstance(value, dict) and "StopIteration" in value:
                    # The sampler returns { "StopIteration":True} to indicate
                    # the end of the dataset. See class _RepeatSampler
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
        self._dataset = dataset

        # Set _worker to None  in case something goes wrong in the AsynchronousWorker constructor
        self._worker = None
        self._worker = _impl.AsynchronousWorker(buffer_size,
                                                miss_sleep_time_in_ms, dataset,
                                                load_indefinitely)

    def terminate(self):
        """
        An override function to kill the worker process manually.
        """
        if self._worker:
            self._worker.terminate()

    def __del__(self):
        self.terminate()

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        self._worker.resetIterator()
        return self

    def __next__(self):
        # We return shared memory to the user so we can't tell the worker to
        # refill it until the next item is requested.
        self._worker.releaseElement()
        while not self._worker.endOfFile():
            data = self._worker.acquireElementIfAvailable()
            if data is not None:
                return data

        self._worker.assertNoError()
        # EOF event
        raise StopIteration


def trainingModel(model, options=None, optimizer=None):
    """ Create a PopTorch training model, from a PyTorch model, to run on IPU
    hardware in training mode.

    :param torch.nn.Module model: The PyTorch model to wrap.
    :param poptorch.Options options: The IPU specific options
    :param torch.optim.Optimizer optimizer: The optimizers to apply during \
        training.

        Supported PyTorch optimizers: ``optim.SGD``, ``optim.Adam``, \
             ``optim.AdamW``, ``optim.RMSprop``.

        Supported PopTorch optimizers: :py:class:`poptorch.optim.SGD`, \
            :py:class:`poptorch.optim.Adam`, \
            :py:class:`poptorch.optim.AdamW`, \
            :py:class:`poptorch.optim.RMSprop`. \
            :py:class:`poptorch.optim.LAMB`.

    :returns: The :py:class:`poptorch.PoplarExecutor` wrapper to use in place
        of ``model``.
    """

    maybe_wrapped_model = model

    if optimizer and len(optimizer.param_groups) > 1:
        maybe_wrapped_model = _impl.OptimizerWrapper(model, optimizer)

    return PoplarExecutor(model=maybe_wrapped_model,
                          options=options,
                          training=True,
                          optimizer=optimizer,
                          user_model=model)


def inferenceModel(model, options=None):
    """Create a PopTorch inference model, from a PyTorch model, to run on IPU
    hardware in inference mode.

    :param torch.nn.Module model: The PyTorch model to wrap.
    :param poptorch.Options options: The IPU specific options
    :returns: The :py:class:`poptorch.PoplarExecutor` wrapper to use in place
        of ``model``.
    """
    return PoplarExecutor(model=model, options=options, training=False)


def ipuHardwareIsAvailable():
    """Indicates whether IPU hardware is available to use.

    :returns: True if physical IPUs are available, False otherwise.
    :rtype: bool
    """
    return poptorch_core.ipuHardwareIsAvailable()


def setLogLevel(level):
    """Changes the volume of messages printed in the console (stdout)

    :param str level:
        * TRACE: Print all messages.
        * DEBUG: Print debug messages and above.
        * INFO: Print info messages and above.
        * WARN: Print warings and errors.
        * ERR:  Print errors only.
        * OFF:  Print nothing.
    """
    _logging.setLogLevel(level)
