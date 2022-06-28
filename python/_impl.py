# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from contextlib import contextmanager
import copy
import copyreg
import fcntl
import hashlib
import os
from functools import partial, wraps
from typing import Dict, Any
import torch

# Do not import any poptorch.* here: it will break the poptorch module
from ._logging import logger
from . import poptorch_core

# A flag to tell the user if the current target is IPU. This is to allow
# divergent IPU/CPU codepaths within one model.
_is_ipu_context = False

# A flag to tell if the dispatch mechanism (or jit tracing) is used to obtain
# a graph.
_dispatch_tracing = False

# Some modules will still work even if the buffer address changes during tracing
BUFFERS_CAN_CHANGE = (
    torch.nn.BatchNorm1d,
    torch.nn.modules.batchnorm.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.modules.batchnorm.BatchNorm3d,
)


class NameScopeHook:
    """ Create a name scope for each operator present in the module.
        The operator name scope will be based on the names appearing in the
        named_modules function from torch.nn.Module..
    """

    def __init__(self, module: 'torch.nn.Module'):
        self.hooks = []
        for name, m in module.named_modules():
            if len(name) > 0:
                self.hooks.append(
                    m.register_forward_pre_hook(
                        partial(self._enter_fn, name=name)))
                self.hooks.append(m.register_forward_hook(self._exit_fn))

    def _enter_fn(self, module, input, name):  # pylint: disable=unused-argument
        torch.ops.poptorch.push_name_scope(name.split(".")[-1])

    def _exit_fn(self, module, input, output):  # pylint: disable=unused-argument
        torch.ops.poptorch.pop_name_scope()

    def remove(self):
        """ Remove all existing hooks related to creating a name scope for
            operators.
        """
        for hook in self.hooks:
            hook.remove()


def createPoptorchError(msg):
    type = "poptorch_py_error"
    error = poptorch_core.Error(f"'{type}': {msg}")
    error.type = type
    error.message = msg
    error.location = ""
    return error


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


def isDispatchTracing() -> bool:
    """ This function returns `True` when executing within the IPUScope.
    The flag is set when entering the scope and turned off when exiting.
    """
    global _dispatch_tracing
    return _dispatch_tracing


def setDispatchTracing(val: bool):
    global _dispatch_tracing
    _dispatch_tracing = val


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
# The _wrapper_registry keeps track of the mapping between user model, parameter,
# buffer types and their corresponding wrapper.

# When an object is copied we want to preserve the Wrapper type: the PopTorch
# wrapper doesn't contain any attribute so it's just a question of updating
# the __class__attribute.
#
# When an object is loaded from file: the wrapper type doesn't exist anymore
# therefore we keep the object unwrapped. (It will be wrapped again when passed
# to poptorch.trainingModel anyway)
_wrapper_registry: Dict[int, Any] = {}
# List of all the wrapper types used by PopTorch.
_wrapper_types = []


def _pickleRestoreWrapperIfPossible(obj):
    wrapperType = _wrapper_registry.get(id(obj))
    if wrapperType:
        obj.__class__ = wrapperType
    return obj


def _pickleUnwrapObject(obj):
    global _wrapper_registry
    wrapperType = obj.__class__
    obj.__class__ = obj.__class__.__bases__[0]
    other = copy.copy(obj)
    _wrapper_registry[id(other)] = wrapperType
    obj.__class__ = wrapperType
    return _pickleRestoreWrapperIfPossible, (other, )


def registerWrapperType(wrapper_type):
    global _wrapper_types
    assert wrapper_type not in _wrapper_types
    _wrapper_types.append(wrapper_type)
    copyreg.pickle(wrapper_type, _pickleUnwrapObject)


def isWrapped(obj):
    global _wrapper_types
    return isinstance(obj, tuple(_wrapper_types))


def unwrapIfWrapped(obj):
    if isWrapped(obj):
        obj.__class__ = obj.__class__.__bases__[0]
    return obj


def traceMethod(label):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self._profiling.tracepoint(label):  # pylint: disable=protected-access
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


def forEachParameterAndBuffer(model, fn):
    for module_name, module in model.named_modules():
        if isinstance(module, BUFFERS_CAN_CHANGE):
            continue

        for name, buff in module.named_buffers(prefix=module_name,
                                               recurse=False):
            fn(name, buff)

    for name, param in model.named_parameters():
        fn(name, param)


def getBufferAndParameterTensors(model):
    tensors = {}

    def fn(name, buff):
        tensors[name] = buff

    forEachParameterAndBuffer(model, fn)
    return tensors
