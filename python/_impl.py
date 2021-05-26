# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import os
from typing import Dict, Any
import torch

# Do not import any poptorch.* here: it will break the poptorch module
from ._logging import logger

# A flag to tell the user if the current target is IPU. This is to allow
# divergent IPU/CPU codepaths within one model.
_is_ipu_context = False


def isRunningOnIpu() -> bool:
    """ This function returns `True` when executing on IPU and `False` when
    executing the model outside IPU scope. This allows for separate
    codepaths to be marked in the model simply by using:

    >>> if poptorch.isRunningOnIpu():
    >>>      # IPU path
    >>> else:
    >>>     # CPU path

        Note this will only apply to code during execution. During model
        creation it will always return `False`.

        :returns: True if running on IPU, otherwise False.
    """
    global _is_ipu_context
    return _is_ipu_context


def setIpuContext(val: bool):
    global _is_ipu_context
    _is_ipu_context = val


def internal_cast(tensor, dtype):
    if dtype in [torch.float, torch.float32]:
        return torch.ops.poptorch.internal_cast(tensor, "FLOAT")

    if dtype in [torch.half, torch.float16]:
        return torch.ops.poptorch.internal_cast(tensor, "FLOAT16")

    raise ValueError(
        'Invalid poptorch.cast target type. Expecting torch.float or torch.half'
    )


def applyOptimizer(optimizer):
    num_groups = len(optimizer.param_groups)
    for index in range(0, num_groups):
        torch.ops.poptorch.optimizer_group(
            index, optimizer.param_groups[index]["params"])


# To understand which variable groups the user wants to apply the
# optimizer to we need to mark them via a wrapper. We do this because
# when we reference the variables in the context of the operation we
# get the corresponding IR value for "free" as part of the trace.
# Otherwise we would need a system to map the variable in the optimizer
# to the variable in the model to the variable in the IR.
class OptimizerWrapper(torch.nn.Module):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        applyOptimizer(self.optimizer)
        return out


@contextmanager
def distributedCacheLock(model, opts):
    """In a distributed environment we only want the model to be compiled once.

    If there is only one process or if the cache is not enabled:
        no need for a lock, early return.
    Otherwise:
        The first process to reach the lock takes it and compiles the model.
            The model will be added to the PopART cache.
        After the first process releases the lock the other ones will grab it
            one at the time and compile the model too (Except that they will
            now all hit the cache).
        The last process to grab / release the lock will delete the file.
        (Each process append a character to the file, so the position in
        the file when acquiring the lock indicates how many processes have
        already successfully compiled the model).
    """
    filename = None
    if opts.Distributed.numProcesses > 1:
        cache = opts._popart.options.get("cachePath", "")  # pylint: disable=protected-access
        if not cache:
            logger.warning(
                "Use poptorch.Options.enableExecutableCaching() to avoid "
                "compiling the model once per process")
        else:
            os.makedirs(cache, exist_ok=True)
            assert os.access(cache, os.W_OK), (f"Cache folder {cache}"
                                               " is not writable")
            filename = os.path.join(
                cache, "%s.lock" %
                hashlib.md5(repr(model).encode("utf-8")).hexdigest())

    # Not distributed mode or the cache is not enabled: do nothing.
    if not filename:
        yield False
        return

    delete_file = False
    try:
        with open(filename, "a+") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                # Add a character to the file
                f.write("0")
                logger.debug(
                    "Executable cache file locked by process %s (pos %d/%d)",
                    opts.Distributed.processId, f.tell(),
                    opts.Distributed.numProcesses)
                delete_file = f.tell() == opts.Distributed.numProcesses
                # Only the first process should compile
                yield f.tell() == 1
            finally:
                logger.debug("Process %s released the cache lock",
                             opts.Distributed.processId)
                fcntl.flock(f, fcntl.LOCK_UN)
    finally:
        if delete_file:
            os.remove(filename)


# The pickle handlers are called in two cases: when an object is copied
# (i.e copy.copy(obj)) or when an object is pickled / serialised.
# In both cases the object is first dumped using pickleUnwrapModel and then
# in the copy case _pickleRestoreWrapperIfPossible() is called immediately after
# to create the new object.
#
# The _wrapper_registry keeps track of the mapping between user model types
# and their corresponding wrapper.

# When an object is copied we want to preserve the Wrapper type: the PopTorch
# wrapper doesn't contain any attribute so it's just a question of updating
# the __class__attribute.
#
# When an object is loaded from file: the wrapper type doesn't exist anymore
# therefore we keep the model unwrapped. (It will be wrapped again when passed
# to poptorch.trainingModel anyway)
_wrapper_registry: Dict[int, Any] = {}


def _pickleRestoreWrapperIfPossible(model):
    wrapperType = _wrapper_registry.get(id(model))
    if wrapperType:
        model.__class__ = wrapperType
    return model


def pickleUnwrapModel(model):
    global _wrapper_registry
    wrapperType = model.__class__
    model.__class__ = model.__class__.__bases__[0]
    other = copy.copy(model)
    _wrapper_registry[id(other)] = wrapperType
    model.__class__ = wrapperType
    return _pickleRestoreWrapperIfPossible, (other, )
