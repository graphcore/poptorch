# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import atexit
import copy
from typing import Any, Callable, Dict, Iterator, Optional, Union
import pickle

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

import poptorch.poptorch_core as poptorch_core  # type: ignore

from . import _impl
from .autocasting import autocast
from .enums import *
from .ops import *
from .options import *
from ._impl import PoplarExecutor, isRunningOnIpu
from . import optim
from . import profiling

__version__ = "@VERSION@-@SNAPSHOT@"


def load(filename: str,
         edit_opts_fn: Optional[Callable[['poptorch.Options'], None]] = None
         ) -> 'poptorch.PoplarExecutor':
    """Load a PopTorch model from a file previously created using
    :py:meth:`~poptorch.PoplarExecutor.compileAndExport`

    :param edit_opts_fn: Function to edit the options before the model
        is restored. For example to attach to a specific IPU device.

    >>> model = poptorch.inferenceModel(model)
    >>> model.compileAndExport("my_model.poptorch")
    ...
    >>> model = poptorch.load("my_model.poptorch")
    >>> model(my_input)
    """
    data, _ = _impl.parsePoptorchData(filename, __version__)
    assert data.model and data.options, (
        f"{filename} is a valid PopTorch file but was created"
        " with 'export_model=False' which means you need to re-create"
        " the PopTorch model using poptorch.inferenceModel or "
        "poptorch.trainingModel then call "
        f"poptorch_model.loadExecutable(\"{filename}\").")
    if edit_opts_fn:
        edit_opts_fn(data.options)
    if data.training:
        executor = trainingModel(data.model, data.options, data.optimizer)
    else:
        executor = inferenceModel(data.model, data.options)
    executor.loadExecutable(filename)
    return executor


class _SubDataset:
    """For distributed execution split the dataset into serial blocks of tensors

    All the tensors used by process 0, followed by all the tensors
    used by process 1, etc.

    [p0, p0, p0, ..., p1, p1, p1, ..., p2,p2, p2]

    If ``shuffle`` is enabled, then the indices in the parent dataset are
    randomised and ``swap_range`` will be called every time a new iterator
    is created in order to make sure all the tensors get used.
    """

    def __init__(self, dataset, opts, step, shuffle):
        num_elts = len(dataset)
        # Note: all the processes must have the same number of batches
        # or it will hang.
        per_proc = step * (num_elts // (step * opts.Distributed.numProcesses))
        self._offset = opts.Distributed.processId * per_proc
        self._base_offset = self._offset
        self._length = min(per_proc, num_elts - self._offset)
        self._dataset = dataset
        self._leftovers = num_elts % per_proc

        self._shuffled_indices = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(opts.random_seed)
            self._shuffled_indices = torch.randperm(num_elts,
                                                    generator=generator)

    def swap_range(self):
        """If there are leftovers in the randomly sampled dataset make sure
        they get included in the next iteration.

        For example if we've got: T = N * B + L
        T = total number of tensors
        N = number of full batches in T
        B = batch size
        L = Number of left over tensors

        First the dataset will return the tensors in [0, T-L]
        after ``swap_range`` was called the dataset will return tensors in
        [L, T]
        """
        if self._base_offset == self._offset:
            self._offset += self._leftovers
        else:
            self._offset = self._base_offset

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        global_index = index + self._offset
        if self._shuffled_indices is not None:
            global_index = self._shuffled_indices[global_index]
        return self._dataset[global_index]


class DataLoader(torch.utils.data.DataLoader):
    """ Thin wrapper around the traditional `torch.utils.data.DataLoader` to
    abstract away some of the batch sizes calculations.

    If this DataLoader is used in a distributed execution environment, it will
    ensure that each process uses a different subset of the dataset, providing
    you first call ``options.randomSeed(N)`` with an integer N which is the same
    across all hosts.
    """

    def __init__(self,
                 options: 'poptorch.Options',
                 dataset: 'torch.utils.data.Dataset',
                 batch_size: int = 1,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 drop_last: bool = True,
                 persistent_workers: Optional[bool] = None,
                 auto_distributed_partitioning: bool = True,
                 mode: 'poptorch.DataLoaderMode' = DataLoaderMode.Sync,
                 async_options: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        :param options: Options that will be used to compile
            and run the model.
        :param dataset: The dataset to get the data from.
        :param batch_size: This is the batch size in the conventional sense
            of being the size that runs through an operation in the model at
            any given time.
        :param shuffle: Whether or not the dataset should be shuffled.
        :param num_workers: Number of worker processes to use to read the
            data.
        :param drop_last: If True and the number of elements in the
            dataset is not a multiple of the combined batch size then the
            incomplete batch at the end will be dropped.
        :param persistent_workers: Re-use workers between
            iterations if True.
            If None (default): enabled if num_workers > 0, disabled otherwise.
        :param auto_distributed_partitioning: If True, partitions the
            dataset for distributed execution automatically. Otherwise, it is
            assumed that partitioning has been handled manually.
        :param mode: If `DataLoaderMode.Async`, uses an
            :py:class:`~poptorch.AsynchronousDataAccessor` to access the
            dataset. If `DataLoaderMode.Sync`, accesses the dataset
            synchronously.
        :param async_options: Options to pass to
            :py:class:`~poptorch.AsynchronousDataAccessor`.
        :param kwargs: Other options to pass to the Torch's DataLoader's
            constructor.
        """
        assert isinstance(options, Options)
        if persistent_workers is None:
            persistent_workers = num_workers > 0

        self._combined_batch_size: Optional[int]
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
        self._swap_range = None

        if self._is_iterable:
            if auto_distributed_partitioning:
                assert options.Distributed.numProcesses == 1, (
                    "auto_distributed_partitioning not supported for"
                    " IterableDataset")
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
            if not drop_last and self._combined_batch_size is not None and \
                num_elts % (self._combined_batch_size *
                            options.Distributed.numProcesses) != 0:
                logger.warning(
                    "The number of elements in the dataset "
                    "(%d) is not divisible by the number of"
                    " elements processed per step (%d)"
                    " and drop_last=False. The last tensor will have "
                    "a batch size of %d. To avoid having to handle "
                    "this special case switch to drop_last=True", num_elts,
                    self._combined_batch_size *
                    options.Distributed.numProcesses,
                    num_elts % (self._combined_batch_size *
                                options.Distributed.numProcesses))

            if options.Distributed.numProcesses > 1:
                if auto_distributed_partitioning:
                    assert not shuffle or options.exists("random_seed"), (
                        "When using auto_distributed_partitioning you must set "
                        "poptorch.Options.randomSeed() to ensure that tensors "
                        "are in the same order in all processes.")
                    assert self._combined_batch_size is not None, (
                        "batch_size=None not allowed when using "
                        "auto_distributed_partitioning.")

                    dataset = _SubDataset(dataset, options,
                                          self._combined_batch_size, shuffle)
                    if shuffle:
                        self._swap_range = dataset.swap_range
        if not self._is_iterable:
            dataset = profiling.Channel("dataset").instrument(
                dataset, "__getitem__")

        rebatched_size = None
        dataset_batch_size = self._combined_batch_size
        if mode == DataLoaderMode.AsyncRebatched:
            mode = DataLoaderMode.Async
            if self._is_iterable:
                # If we're rebatching then we force the dataset to use a
                # batch size of 1 and the AsynchronousDataAccessor will
                # build the batched tensor.
                rebatched_size = self._combined_batch_size
                dataset_batch_size = 1
        super().__init__(dataset,
                         batch_size=dataset_batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         drop_last=drop_last,
                         persistent_workers=persistent_workers,
                         **kwargs)

        self._accessor = None
        if mode == DataLoaderMode.Async:
            async_options = async_options or {}
            assert "rebatched_size" not in async_options, (
                "You cannot "
                "use DataLoaderMode.AsyncRebatched and manually specify"
                " the rebatched_size in async_options")
            self._accessor = AsynchronousDataAccessor(
                self, **async_options, rebatched_size=rebatched_size)

    def __len__(self) -> int:
        # If we're rebatching in the AsynchronousDataAccessor we need to
        # adjust the dataset's length.
        dataset_len = super().__len__()
        if self._accessor is not None and self._accessor.rebatched_size:
            if not self.drop_last:
                # Round up
                dataset_len += self._accessor.rebatched_size - 1
            dataset_len = dataset_len // self._accessor.rebatched_size
        return dataset_len

    @property
    def _profiling(self):
        return profiling.Channel("poptorch.DataLoader")

    @property
    def combinedBatchSize(self) -> Optional[int]:
        """Total number of elements consumed from the dataset for a single
        execution of the model."""
        return self._combined_batch_size

    @property
    def options(self) -> 'poptorch.Options':
        """A reference to the options that were used to initialise this
        DataLoader.
        """
        return self._options

    def terminate(self) -> None:
        """If `mode==DataLoaderMode.Async`, kills the worker process in the
        underlying AsynchronousDataAccessor manually, otherwise has no effect.
        """
        if self._accessor is not None:
            self._accessor.terminate()

    def __del__(self) -> None:
        self.terminate()

    def __iter__(self) -> "torch.utils.data.dataloader._BaseDataLoaderIter":
        if self._swap_range is not None:
            self._swap_range()
        if self._accessor is not None:
            return self._accessor.__iter__()

        return super().__iter__()


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

    .. note:: When using a ``torch.utils.data.Dataset`` with ``rebatched_size``
        the accessor will default to ``drop_last=True``, to change that
        behaviour wrap the dataset into a
        ``poptorch.DataLoader(..., drop_last=False)``.
    """

    def __init__(
            self,
            dataset: Union['torch.utils.data.Dataset', DataLoader],
            buffer_size: int = 3,
            miss_sleep_time_in_ms: float = 0.1,
            load_indefinitely: bool = True,
            early_preload: bool = False,
            sharing_strategy: 'poptorch.SharingStrategy' = SharingStrategy.
            FileSystem,
            rebatched_size: Optional[int] = None):
        """
        :param dataset: The dataset to pull data from, this can be any Python
            iterable.
        :param buffer_size: The size of the ring buffer.
        :param miss_sleep_time_in_ms: When the buffer is full how long should
            we sleep the worker before checking again.
        :param load_indefinitely: If True when we hit the end of the dataset
            we will just loop round again.
        :param early_preload: If True, start loading data in the ring buffer
            as soon as the worker is created.
            If False, wait for an iterator to be created before loading data.
        :param sharing_strategy: Method to use to pass the dataset object when
            the child process is spawned.
            SharedMemory is fast but might be quite limited in size.
            FileSystem will serialise the dataset to file and reload it which
            will be slower.
        :param rebatched_size: If not None: return N batched tensors from
            the dataset per iteration. (The passed dataset must have a
            batch_size of 1).

        .. note :: If dataset is an iterable-type ``poptorch.DataLoader``
            configured with ``drop_last=False`` then ``rebatched_size``
            must be used.
        """

        # Ensure the DataLoader doesn't already have an AsynchronousDataAccessor
        if isinstance(dataset, DataLoader) and dataset._accessor is not None:
            raise RuntimeError(
                "The DataLoader already uses an "
                "AsynchronousDataAccessor internally. Either use "
                "the existing one or set mode='poptorch.DataLoaderMode.Sync'"
                " in the DataLoader.")

        if isinstance(dataset, DataLoader) and not dataset.drop_last and \
                rebatched_size is None and dataset._is_iterable:
            # Otherwise we'll end up with one left over tensor per worker
            # to return to the main process and we don't currently
            # support that.
            assert dataset.combinedBatchSize is None or \
                   dataset.combinedBatchSize == 1, (
                       "The 'drop_last=False' option from the DataLoader only "
                       "works if 'rebatched_size' is specified too.")
        if isinstance(dataset, DataLoader
                      ) and not dataset.drop_last and not dataset._is_iterable:
            # We don't currently support returning a partial batch through the ring buffer
            # for map-style datasets
            assert dataset.combinedBatchSize is None or \
                   dataset.combinedBatchSize == 1, (
                       "The 'drop_last=False' option from the DataLoader is"
                       " not currently supported by the "
                       "AsynchronousDataAccessor for map-style datasets")
        if rebatched_size is not None:
            assert rebatched_size > 1, ("rebatched_size"
                                        " must be None or greater than 1")

        self._dataset = dataset
        # Set _worker to None  in case something goes wrong in the AsynchronousWorker constructor
        self._worker = None
        # To avoid hangs when the application exits: implicitly call terminate().
        atexit.register(self.terminate)
        self.rebatched_size = rebatched_size
        self._worker = _impl.AsynchronousWorker(
            buffer_size, miss_sleep_time_in_ms, dataset, load_indefinitely,
            early_preload, sharing_strategy, rebatched_size)

    def terminate(self) -> None:
        """
        An override function to kill the worker process manually.
        """
        if self._worker:
            self._worker.terminate()

    def __del__(self) -> None:
        self.terminate()

    def __len__(self) -> int:
        dataset_len = len(self._dataset)
        # If this AsynchronousDataAccessor is embedded in a DataLoader then the dataset
        # length has already been adjusted.
        if self.rebatched_size and getattr(self._dataset, "_accessor",
                                           None) != self:
            dataset_len = dataset_len // self.rebatched_size
        return dataset_len

    def __iter__(self) -> 'poptorch.AsynchronousDataAccessor':
        assert self._worker
        self._worker.resetIterator()
        return self

    def __next__(self) -> Any:
        # We return shared memory to the user so we can't tell the worker to
        # refill it until the next item is requested.
        assert self._worker
        self._worker.releaseElement()
        while not self._worker.endOfFile():
            data = self._worker.acquireElementIfAvailable()
            if data is not None:
                return data
            self._worker.assertNoError()
        # EOF event
        raise StopIteration


def trainingModel(model: Union['torch.nn.Module', 'poptorch.PoplarExecutor'],
                  options: Optional['poptorch.Options'] = None,
                  optimizer: Optional['torch.optim.Optimizer'] = None
                  ) -> 'poptorch.PoplarExecutor':
    """ Create a PopTorch training model, from a PyTorch model, to run on IPU
    hardware in training mode.

    :param model: The PyTorch model to wrap.
    :param options: The IPU specific options
    :param optimizer: The optimizers to apply during \
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
    if isinstance(model, PoplarExecutor):
        model = model._user_model  # pylint: disable=protected-access

    # Create a copy of the original model in case it needs to be wrapped
    maybe_wrapped_model = copy.copy(model)

    return PoplarExecutor(model=maybe_wrapped_model,
                          options=options,
                          training=True,
                          optimizer=optimizer,
                          user_model=model,
                          poptorch_version=__version__)


def inferenceModel(model: Union['torch.nn.Module', 'poptorch.PoplarExecutor'],
                   options: Optional['poptorch.Options'] = None
                   ) -> 'poptorch.PoplarExecutor':
    """Create a PopTorch inference model, from a PyTorch model, to run on IPU
    hardware in inference mode.

    :param model: The PyTorch model to wrap.
    :param options: The IPU specific options
    :returns: The :py:class:`poptorch.PoplarExecutor` wrapper to use in place
        of ``model``.
    """
    if isinstance(model, PoplarExecutor):
        model = model._user_model  # pylint: disable=protected-access
    return PoplarExecutor(model=copy.copy(model),
                          options=options,
                          training=False,
                          poptorch_version=__version__)


def ipuHardwareIsAvailable(num_ipus: int = 1) -> bool:
    """Indicates whether any IPU hardware with `num_ipus` is present in the system.

    Note: This function doesn't check if the IPU is free or already being used.

    :param num_ipus:
    :returns: True if physical IPUs are available, False otherwise.
    """
    return poptorch_core.ipuHardwareVersion(num_ipus) != 0


def ipuHardwareVersion() -> int:
    """Indicates what IPU hardware version is available in the system.

    Raise an exception if no hardware is available.

    :returns: The IPU hardware version or -1 if unknown.
    """
    version = poptorch_core.ipuHardwareVersion()
    assert version != 0, "No IPU hardware available on this system"
    return version


def setLogLevel(level: Union[str, int]):
    """Changes the volume of messages printed in the console (stdout)

    :param level:
        * TRACE: Print all messages.
        * DEBUG: Print debug messages and above.
        * INFO: Print info messages and above.
        * WARN: Print warings and errors.
        * ERR:  Print errors only.
        * OFF:  Print nothing.
    """
    _logging.setLogLevel(level)
