# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from contextlib import contextmanager
import copy
import copyreg
import fcntl
import hashlib
import itertools
import os
from functools import partial, wraps
import weakref
import torch

# Do not import any poptorch.* here: it will break the poptorch module
from ._logging import logger
from . import poptorch_core
from ._utils import isOnIpu, getIpuTensorId

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


def destroyDispatcherOnExit(func):
    """Function decorator to always destroy the dispatcher at
    the end of the wrapped function."""

    class OnExit():
        def __enter__(self):
            pass

        def __exit__(self, exc_type, value, traceback):
            poptorch_core.destroyDispatcher()

    @wraps(func)
    def wrapper(*args, **kwargs):
        with OnExit():
            return func(*args, **kwargs)

    return wrapper


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


# A helper class that compares using pointer semantics rather than value
# semantics (i.e. comparing using `is` rather than eq). This is needed because
# Tensor comparison in torch returns a Tensor rather than an boolean
class WeakPtr(weakref.ref):
    __hash__ = weakref.ref.__hash__

    def __eq__(self, other):
        s = self()
        o = other()
        return self is other if s is None else s is o


# Our own dictionary with weak keys that compares keys using pointer semantics
# rather than value semantics (i.e. comparing using `is` rather than `eq`). We
# use this rather than a weakref.WeakKeyDictionary because that uses equality on
# values to compare items.
#
# Note: that we do not provide functionality for iterating over the dictionary
# since there will be issues if the cleanup function is called while iterating
class WeakKeyPtrDict:
    def __init__(self, dict=None):
        self.data = {}

        def cleanup(k, selfref=weakref.ref(self)):
            self = selfref()
            if self is not None:
                del self.data[k]

        self._cleanup = cleanup

        self.update(dict)

    def __setitem__(self, key, value):
        self.data[WeakPtr(key, self._cleanup)] = value

    def __delitem__(self, key):
        del self.data[WeakPtr(key)]

    def __getitem__(self, key):
        return self.data[WeakPtr(key)]

    def get(self, key, default=None):
        return self.data.get(WeakPtr(key), default)

    def __contains__(self, key):
        return WeakPtr(key) in self.data

    def update(self, dict=None):
        if dict is not None:
            for k, v in dict.items():
                self.__setitem__(k, v)


# The pickle handlers are called in two cases: when an object is copied
# (i.e copy.copy(obj)) or when an object is pickled / serialised.
# In both cases the object is first dumped using pickleUnwrapModel and then
# in the copy case _restoreWrapperIfNecessary() is called immediately after
# to create the new object.
#
# The _wrapper_registry keeps track of the mapping between user model, parameter,
# buffer types and their corresponding wrapper.

# When an object is copied we want to preserve the Wrapper type: the PopTorch
# wrapper doesn't contain any attribute so it's just a question of updating
# the __class__attribute.
#
# When an object is loaded from file: the wrapper type doesn't exist any more
# therefore we keep the object unwrapped. (It will be wrapped again when passed
# to poptorch.trainingModel anyway)
_wrapper_registry = WeakKeyPtrDict()
# List of all the wrapper types used by PopTorch.
_wrapper_types = []


def _restoreWrapperIfNecessary(obj):
    wrapperType = _wrapper_registry.get(obj)
    if not wrapperType is None:
        obj.__class__ = wrapperType
    return obj


def _unwrapIfWrappedAndRegister(obj):
    global _wrapper_registry
    if isWrapped(obj):
        wrapperType = obj.__class__
        obj.__class__ = obj.__class__.__bases__[0]
        _wrapper_registry[obj] = wrapperType


def _pickleUnwrapObject(obj):
    global _wrapper_registry
    wrapperType = obj.__class__
    if not wrapperType in _wrapper_types:
        raise createPoptorchError("Internal Error")

    # We need to unwrap obj before copying it because this is the function
    # registered for doing copies
    obj.__class__ = obj.__class__.__bases__[0]
    other = copy.copy(obj)
    _wrapper_registry[other] = wrapperType
    obj.__class__ = wrapperType
    return _restoreWrapperIfNecessary, (other, )


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


def unwrapModelIfNecessary(model: torch.nn.Module):
    # Removes the PoptorchParameter and PoptorchBuffer annotations in the model

    for buff in itertools.chain(model.buffers(), model.parameters()):
        _unwrapIfWrappedAndRegister(buff)


def rewrapModelIfNecessary(model: torch.nn.Module):
    # Restores the PoptorchParameter and PoptorchBuffer annotations in the model

    for buff in itertools.chain(model.buffers(), model.parameters()):
        _restoreWrapperIfNecessary(buff)


def getBufferAndParameterTensors(model):
    tensors = {}

    def fn(name, buff):
        tensors[name] = buff

    forEachParameterAndBuffer(model, fn)
    return tensors


def getBufferAndParameterAddresses(model):
    # Obtains dictionaries of the data ptr addresses of every buffer
    # and parameter

    def tensor_info(x):
        if isOnIpu(x):
            return x.device, getIpuTensorId(x)
        return x.device, x.data_ptr()

    buffer_addresses = {}
    for module_name, module in model.named_modules():
        if isinstance(module, BUFFERS_CAN_CHANGE):
            continue

        for name, buff in module.named_buffers(prefix=module_name,
                                               recurse=False):
            buffer_addresses[name] = tensor_info(buff)

    parameter_addresses = {}
    for name, param in model.named_parameters():
        parameter_addresses[name] = tensor_info(param)

    return buffer_addresses, parameter_addresses


def errorOnBufferOrParameterAddressChanges(old_addresses, new_addresses):
    # Do the buffers first then parameters
    order = ["Buffer", "Parameter"]
    for idx, dic in enumerate(old_addresses):
        for name, address in dic.items():
            if name not in new_addresses[idx]:
                err_msg = (order[idx] + " " + name + " is removed from " +
                           "the model when calling the forward method.")

                raise createPoptorchError(err_msg)

            if address != new_addresses[idx][name]:
                err_msg = (
                    order[idx] + " " + name + " is reassigned " +
                    "within the model when calling the forward " +
                    "method. This is not supported. Consider using self." +
                    name + ".copy_(src)" +
                    " to copy data from a source tensor, where src is " +
                    "the name of the source tensor.")
                raise createPoptorchError(err_msg)


# Wrapper to make sure that the buffers and parameters do not change during
# tracing (which would give wrong results in a Jit trace)
class CheckBuffersAndParamsScope:
    def __init__(self, model: 'torch.nn.Module'):
        self._model = model
        self._old_addresses = {}

    def __enter__(self):
        if self._model:
            self._old_addresses = getBufferAndParameterAddresses(self._model)

    def __exit__(self, exc_type, value, traceback):
        if self._model:
            new_addresses = getBufferAndParameterAddresses(self._model)
            errorOnBufferOrParameterAddressChanges(self._old_addresses,
                                                   new_addresses)
