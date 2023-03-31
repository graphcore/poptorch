# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import collections
import copy
import functools
import itertools
import os
import pickle
from typing import Callable, Dict, List, Optional
from types import MethodType
import weakref
import warnings
import torch

# Do not import any poptorch.* here: it will break the poptorch module
from . import _impl
from . import _utils
from . import _args_parser
from . import _optimizer_attributes
from . import enums
from . import _printing
from . import optim
from . import profiling
from . import poptorch_core  # type: ignore
from . import _poptorch_data
from ._utils import accessAttributes, flattenTensorStructure, reconstructTensorStructure, isOnIpu
from ._logging import logger
from .options import Options, PipelinedExecution, ShardedExecution
from .optim import Optimizer
NO_EXECUTABLE_ERR = "Model has not been compiled or has been destroyed."


# Hacky way to make sure tensors end up on the IPU rather than the CPU by default.
# Note: this is only needed for backward compatibility with tracing but we will
# eventually stop supporting this approach so make sure a warning is printed.
class _SetDefaultDeviceType:
    def __init__(self):
        self.overrides = dict()
        self.saved_distribution_validate_args = None

    def replace(self):
        def create_wrapper(f):
            @functools.wraps(f)
            def _wrapper(*args, **kwargs):
                if "device" not in kwargs:
                    logger.warning(
                        "No device set in torch.%s(): forcing to IPU",
                        f.__name__)
                    kwargs["device"] = "ipu"
                return f(*args, **kwargs)

            return _wrapper

        # All the ops with FACTORY_PARAMS in <torch>/tools/pyi/gen_pyi.py
        for name in ("arange", "empty", "full", "full_like", "linspace",
                     "logspace", "ones", "rand", "randint", "randn",
                     "randperm", "range", "tensor", "zeros", "zeros_like"):
            func = getattr(torch, name)

            self.overrides[name] = func
            setattr(torch, name, create_wrapper(func))

        def create_non_tensor_wrapper(f):
            @functools.wraps(f)
            def _wrapper(*args, **kwargs):
                if not any(
                        isinstance(a, torch.Tensor) for a in itertools.chain(
                            args, kwargs.values())) and "device" not in kwargs:
                    logger.warning(
                        "No device set in torch.%s(): forcing to IPU",
                        f.__name__)
                    kwargs["device"] = "ipu"
                return f(*args, **kwargs)

            return _wrapper

        # overloaded ops that take a device for some overloads
        for name in ["normal"]:
            func = getattr(torch, name)

            self.overrides[name] = func
            setattr(torch, name, create_non_tensor_wrapper(func))

        # Arguments validation forces the tensors to be compared onto the IPU
        # then the result is sent back to the CPU.
        # For example:
        # >>> if self._validate_args:
        # >>>    assert torch.lt(self.low, self.high).all()
        # pylint: disable=protected-access
        self.saved_distribution_validate_args = \
            torch.distributions.Distribution._validate_args
        torch.distributions.Distribution.set_default_validate_args(False)

    def restore(self):
        # Restore the real Torch functions
        for name, real in self.overrides.items():
            setattr(torch, name, real)

        torch.distributions.Distribution.set_default_validate_args(
            self.saved_distribution_validate_args)


class _OverwriteContextManager:

    _subsitution_manager_types = [_SetDefaultDeviceType]

    def __init__(self):
        self.substitution_managers = [
            manager_type() for manager_type in
            _OverwriteContextManager._subsitution_manager_types
        ]

    def __enter__(self):
        for overwriter in self.substitution_managers:
            overwriter.replace()

        return self

    def __exit__(self, exc_type, value, traceback):
        for overwriter in reversed(self.substitution_managers):
            overwriter.restore()

    @classmethod
    def registerSubsitutionManager(cls, type):
        if type not in cls._subsitution_manager_types:
            cls._subsitution_manager_types.append(type)


# pylint: disable=too-many-public-methods
class PoplarExecutor:
    """ This class should not be created directly but is a wrapper around
    the model that was passed into `inferenceModel` or `trainingModel`.
    It only has a few methods which can be used to interface with the IPU.
    """

    _precompile_hooks: Dict[int, Callable] = collections.OrderedDict()
    _postcompile_hooks: Dict[int, Callable] = collections.OrderedDict()

    # pylint: disable=too-many-statements
    def __init__(self,
                 model: 'torch.nn.Module',
                 options: Optional['poptorch.Options'],
                 training: bool,
                 poptorch_version: str,
                 optimizer: Optional['torch.optim.Optimizer'] = None,
                 user_model: Optional['torch.nn.Module'] = None):
        if options:
            if not isinstance(options, Options):
                raise _impl.createPoptorchError(
                    "Invalid type: 'options' is of "
                    f"type {type(options)} (Expected poptorch.Options)")
            # Prevent the user from modifying these options.
            options._freeze()
            options = options.clone()
        else:
            options = Options()

        # NB model is the one which gets called, which may have its own wrapping
        # such as to have a loss. user_model is the one which is transformed.

        self._user_model = user_model or model

        if training:
            self._attribute_tracker = \
                    _optimizer_attributes.OptimizerAttrTracker(
                        options)
            if options.defaultOutputMode():
                # In training it makes sense to see only the last result, by
                # default.
                options.outputMode(enums.OutputMode.Final)
            if not optimizer:
                optimizer = optim.SGD(self._user_model.parameters(), lr=0.01)
        else:
            if options.defaultOutputMode():
                # In inference it makes sense to see all the results, by default.
                options.outputMode(enums.OutputMode.All)

            if options.Training.gradient_accumulation != 1:
                err_msg = (
                    "You must set "
                    "poptorch.Options().Training.gradientAccumulation to 1 "
                    "or leave it as its default value (1) when running a "
                    "poptorch.inferenceModel().")

                is_pipelined = (isinstance(options._execution_strategy,
                                           PipelinedExecution)
                                and not isinstance(options._execution_strategy,
                                                   ShardedExecution))
                if is_pipelined:
                    err_msg += (" Use poptorch.Options().deviceIterations "
                                "to process a sufficient number of batches "
                                "each run for pipelined execution instead.")

                raise _impl.createPoptorchError(err_msg)

            assert options.Training.gradient_accumulation == 1, ()
            assert not optimizer, "Optimizer should be None for inference"
        self._model = model

        self._host_weights_version = 0
        self._poptorch_version = poptorch_version

        self._executable = None
        self._outputs_structure = None
        self._options = options
        # The args parser needs to be initilialised before the model gets wrapped
        # otherwise we will not be able to retrieve the real arguments list
        self._args_parser = _args_parser.ArgsParser(model)
        # Inputs used to compile the executable
        self._executable_inputs = None
        self._anchor_memory = {}

        # any anchors with unspecified output mode should receive the output
        # mode used for graph outputs
        for _, anchor in options.anchored_tensors.items():
            if anchor[1]:
                anchor[2] = options.output_mode
                if anchor[2] == enums.OutputMode.EveryN:
                    anchor[3] = options.output_return_period

        self._optimizer = optimizer
        self._ipu_optimizer_is_dirty = False
        self._host_rng_state_is_dirty = False
        self._cached_rng_state = None
        if self._options.exists("random_seed"):
            self._cached_rng_state = [self._options.random_seed]

        self._dict_optimizer = {}
        self.per_replica_params = {}
        self._training = training
        self._dirty_host_weights = False
        self._trace = None
        self._is_attached = False
        self._profiling = profiling.Channel(
            "poptorch.trainingModel" if self.
            training else "poptorch.inferenceModel")
        self._profiling.instrument(self, "copyWeightsToHost",
                                   "copyWeightsToDevice", "setOptimizer",
                                   "compile", "destroy")

        if optimizer:
            self.setOptimizer(optimizer)
        self._options._freeze()

        if self._training:
            # We don't want the pytorch model to keep the PopTorch one
            # alive so only keep a weak reference.
            parent = weakref.ref(self)

            class PoptorchModel(type(self._user_model)):
                def copyWeightsToHostIfNeeded(self):
                    """ Return True if the weights on the host were dirty and
                    have been updated.
                    Return False if the weights were already up to date.
                    """
                    p = parent()
                    if p is not None:
                        return p.copyWeightsToHostIfNeeded()
                    return False

                def destroy(self):
                    """Destroy the model: release the IPUs and the executable.
                    """
                    p = parent()
                    if p is not None:
                        p.destroy()

                def __getattribute__(self, name):
                    if name == "_host_weights_version":
                        p = parent()
                        if p is None:
                            return None
                        return p._host_weights_version
                    if name in ("_buffers", "_parameters", "forward"):
                        self.copyWeightsToHostIfNeeded()
                    return object.__getattribute__(self, name)

                def __getattr__(self, name):
                    attribute = super().__getattr__(name)
                    if isinstance(attribute, torch.nn.parameter.Parameter):
                        self.copyWeightsToHostIfNeeded()
                    return attribute

                def state_dict(self,
                               *args,
                               destination=None,
                               prefix="",
                               keep_vars=False):
                    """Return a shallow copy of the wrapped model's state dictionary.

                    Note: all the elements in the state dictionary are
                    unwrapped which means the state can be reloaded in an
                    environment where PopTorch is not installed.
                    """
                    out = collections.OrderedDict()
                    out_cache = {}

                    for k, v in super().state_dict(*args, destination, prefix,
                                                   keep_vars).items():
                        v_id = id(v)

                        # If the value occurs more than once, avoid multiple
                        # copies.
                        if v_id in out_cache:
                            out[k] = out_cache[v_id]
                        else:
                            # If the object is wrapped then the shallow copy will
                            # call _impl._pickleUnwrapObject and the new object will be in
                            # the wrapped registry.
                            # Unwrap the object if needed.
                            v_copy = _impl.unwrapIfWrapped(copy.copy(v))
                            out[k] = v_copy
                            out_cache[v_id] = v_copy

                    return out

            _utils.assert_signatures_match(PoptorchModel.state_dict,
                                           torch.nn.Module.state_dict)

            # The mere existence of the "__torch_function__" results in a
            # "__getattribute__" call and hence weight copying if required.
            # "check_has_torch_function" and "handle_torch_function_getter"
            # in the Pytorch source code may explain this.
            # Without this, the weights will not be copied in certain
            # situations such as torch.equal(a, b).
            class PoptorchParameter(torch.nn.Parameter):
                def __getattribute__(self, name):
                    p = parent()
                    if p is not None:
                        p.copyWeightsToHostIfNeeded()

                    return object.__getattribute__(self, name)

                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    if kwargs is None:
                        kwargs = {}
                    return super().__torch_function__(func, types, args,
                                                      kwargs)

            self.PoptorchParameter = PoptorchParameter

            class PoptorchBuffer(torch.Tensor):
                def __getattribute__(self, name):
                    p = parent()
                    if p is not None:
                        p.copyWeightsToHostIfNeeded()

                    return super().__getattribute__(name)

                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    if kwargs is None:
                        kwargs = {}
                    return super().__torch_function__(func, types, args,
                                                      kwargs)

            self.PoptorchBuffer = PoptorchBuffer
            self._install_state_hooks()

            # __getattr__ and __getattribute__ are attributes, not methods,
            # unfortunately we cannot just replace them in the model object: we
            # have to create a wrapper class
            # and change the object's class.
            PoptorchModel.__name__ = "Poptorch%s" % type(
                self._user_model).__name__
            self._user_model.__class__ = PoptorchModel

            # Register the wrapper types so that custom functions to
            # copy / serialize wrapped objects are set up.
            _impl.registerWrapperType(PoptorchModel)
            _impl.registerWrapperType(PoptorchParameter)
            _impl.registerWrapperType(PoptorchBuffer)

    def _install_state_hooks(self):
        for p in self._user_model.parameters():
            p.__class__ = self.PoptorchParameter
        for b in self._user_model.buffers():
            if not b.__class__ in (torch.Tensor, self.PoptorchBuffer):
                raise _impl.createPoptorchError(
                    "All buffers must be an instance of torch.Tensor "
                    f"(Got {type(b)})")
            b.__class__ = self.PoptorchBuffer

    def _update_optimizer_if_needed(self):
        if not self.isCompiled():
            raise _impl.createPoptorchError(NO_EXECUTABLE_ERR)
        if self._ipu_optimizer_is_dirty:
            poptorch_core.updateOptimizers(self._executable,
                                           self._dict_optimizer)
            self._ipu_optimizer_is_dirty = False

    def _read_optim_state_dict_if_needed(self):
        if not isinstance(self._optimizer, Optimizer):
            return
        if self._optimizer.host_state_is_dirty:
            if not self.isCompiled():
                raise _impl.createPoptorchError(NO_EXECUTABLE_ERR)
            assert not self._ipu_optimizer_is_dirty, (
                "Both host "
                "and ipu states cannot be dirty at the same time.")

            # We need to return both the internal state dict and torch's
            # state dict so that LR schedulers work
            self._optimizer.set_state_dict({
                **poptorch_core.readOptimizerState(self._executable),
                **torch.optim.Optimizer.state_dict(self._optimizer)
            })
            # Don't trigger a copy to IPU as we've just synced.
            self._optimizer.ipu_state_is_dirty = False
        else:
            logger.debug("Using cached optimiser state dict")

    def _on_device_attach(self):
        """Method called every time we attach to a device."""
        # Upload the weights to the IPU
        self.copyWeightsToDevice()
        # Upload the optimizer parameters
        if self._optimizer:
            self._update_optimizer_if_needed()
        # If the optimizer has a state: restore it.
        if self._optimizer and isinstance(self._optimizer, Optimizer):
            # If the optimiser has state to be written (from a checkpoint),
            # write it immediately after compilation
            if self._optimizer.has_state():
                self._optimizer.ipu_state_is_dirty = True
                self._write_optim_state_dict_if_needed()
            else:
                self._optimizer.host_state_is_dirty = True
                self._optimizer.ipu_state_is_dirty = False
        if self._cached_rng_state is not None:
            self._copyRngStateToDevice()

    def _get_optim_state_dict(self):
        assert isinstance(self._optimizer, Optimizer)
        self._read_optim_state_dict_if_needed()
        return self._optimizer.get_state_dict()

    def _write_optim_state_dict_if_needed(self):
        if not self.isCompiled():
            raise _impl.createPoptorchError(NO_EXECUTABLE_ERR)
        # If the new optimiser already has state (i.e. from a checkpoint), write it
        # to device
        if isinstance(self._optimizer,
                      Optimizer) and self._optimizer.ipu_state_is_dirty:
            assert not self._optimizer.host_state_is_dirty, (
                "Both host "
                "and ipu states cannot be dirty at the same time.")
            if self._optimizer.has_state():
                # Sync the weights to host first because writeOptimizerState() is
                # going to write both the weights and the optimizer state
                self.copyWeightsToHostIfNeeded()

                poptorch_core.writeOptimizerState(self._executable,
                                                  self._optimizer.state_dict())
            self._optimizer.ipu_state_is_dirty = False

    def load_state_dict(self,
                        state_dict: Dict[str, 'torch.Tensor'],
                        strict: bool = True):
        """Will call ``load_state_dict()`` on the wrapped model
        and automatically synchronise the weights with the IPU.

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the
                    unexpected keys
        """
        out = self._user_model.load_state_dict(state_dict, strict)
        if self.isAttachedToDevice():
            logger.debug("load_state_dict: implicit copyWeightsToDevice()")
            self.copyWeightsToDevice()
        return out

    def __repr__(self):
        # We've created out repr function to provide info on BeginBlock
        return _printing.module_repr(self._user_model)

    def __getattr__(self, attr):
        model_attr = getattr(self._user_model, attr)
        # We apply this wrapper here rather than adding it to PoptorchParameter
        # for two reasons:
        # 1) We might supply the same model to multiple PopTorch wrappers
        #    (particularly we might supply it to trainingModel() and then
        #     to inferenceModel()), and we need to be able to distinguish
        #     between replicaGrouping() calls on each wrapper.
        # 2) We don't wrap inference parameters in PoptorchParameter normally,
        #    but we might want to use replicaGrouping() with inference models.
        #    If we do start doing PoptorchParameter wraps on inference models,
        #    we'd end up pointlessly copying weights back from the device.
        if isinstance(model_attr, torch.nn.Parameter):
            model = self

            class ReplicaGroupingWrapper:
                def replicaGrouping(
                        self, comm_group_type: enums.CommGroupType,
                        shards: int,
                        variable_retrieval_mode: enums.VariableRetrievalMode):
                    model.per_replica_params[attr] = (comm_group_type, shards,
                                                      variable_retrieval_mode)

                def __getattr__(self, attr):
                    if attr == "replicaGrouping":
                        return self.replicaGrouping
                    return getattr(model_attr, attr)

            return ReplicaGroupingWrapper()
        return model_attr

    @property
    def model(self) -> 'torch.nn.Module':
        """Access the wrapped Torch model."""
        return self._user_model

    @property
    def options(self) -> 'poptorch.Options':
        """Access to the options.

        .. seealso:: :py:class:`~poptorch.Options`"""
        return self._options

    def _debugGetPopartIR(self) -> str:
        return poptorch_core._getPopartIR(self._executable)  # pylint: disable=protected-access

    def getTensorNames(self) -> List[str]:
        """Returns a list of all tensor names within the computational
        graph. Model must be compiled in advance.
        """

        assert self._executable is not None, "Model must be compiled " \
            "before calling getTensorNames"

        tensor_names = poptorch_core._getTensorNames(self._executable)  # pylint: disable=protected-access

        return list(tensor_names)

    def getAnchoredTensor(self, short_name: str) -> torch.Tensor:
        assert short_name in self._anchor_memory, \
            "No tensor with name " + short_name + " found."
        return self._anchor_memory[short_name]

    def copyWeightsToHostIfNeeded(self) -> bool:
        """ Return True if the weights on the host were dirty and
        have been updated.
        Return False if the weights were already up to date.
        """
        if self._dirty_host_weights:
            logger.debug("Implicit copyWeightsToHost()")
            self.copyWeightsToHost()
            return True
        return False

    # Copy weights from the device into the memory of the model given on wrapper creation.
    def copyWeightsToHost(self) -> None:
        """ Updates the parameters used in `model` with the weights stored on device.
        (The weights in ``model.parameters()``)
        """

        if not self.isCompiled():
            raise _impl.createPoptorchError(NO_EXECUTABLE_ERR)

        # Don't trigger another copyToHost by accessing `named_parameters`
        self._dirty_host_weights = False

        weights = {
            **dict(self._model.named_parameters()),
            **dict(self._model.named_buffers())
        }
        poptorch_core.copyWeightsToHost_impl(self._executable,
                                             tuple(weights.keys()),
                                             tuple(weights.values()))
        self._host_weights_version += 1

    # Write from host memory to IPU memory. This is done automatically on
    # compilation so should be rarely used.
    def copyWeightsToDevice(self) -> None:
        """Copies the weights from ``model.parameters()`` to the IPU device.
        Implicitly called on first call.
        """
        if not self.isCompiled():
            raise _impl.createPoptorchError(NO_EXECUTABLE_ERR)

        # Don't trigger a copyToHost by accessing `named_parameters`
        self._dirty_host_weights = False

        # Trigger a IPU sync -> host if needed for
        # the optimizer state.
        if self._optimizer:
            self._optimizer.state_dict()

        weights = {
            **dict(self._model.named_parameters()),
            **dict(self._model.named_buffers())
        }
        poptorch_core.copyWeightsToDevice_impl(self._executable,
                                               tuple(weights.keys()),
                                               tuple(weights.values()))

    def setOptimizer(self, optimizer: 'torch.optim.Optimizer'):
        """Sets the optimiser for a training model. Will overwrite the
        previous one. Supported optimisers: ``optim.SGD``, ``optim.Adam``,
        ``optim.AdamW``, ``optim.RMSProp``, ``optim.LAMB``.
        """
        # Optimiser state functions require a compiled executable
        if self.isCompiled() and optimizer != self._optimizer:
            # If we're setting a new optimiser, make sure the internal state of the old
            # optimiser has been read back so it's not lost, and then detach the old
            # optimiser so that its subsequent state_dict/load_state_dict calls don't
            # trigger optimiser state read/writes anymore
            if self._optimizer and isinstance(self._optimizer, Optimizer):
                self._read_optim_state_dict_if_needed()
                self._optimizer.state_dict = \
                self._optimizer.get_state_dict
            # We only want to update the state on the IPU if it's a brand new optimizer
            # (Not if the params of the existing one have changed).
            if isinstance(optimizer, Optimizer):
                optimizer.ipu_state_is_dirty = True

        # If it's a PopTorch optimizer: instrument the state_dict() method
        # to implicitly transfer the state back to the host.
        if isinstance(optimizer, Optimizer):
            optimizer.state_dict = MethodType(
                PoplarExecutor._get_optim_state_dict, self)

        self._optimizer = optimizer
        dict_optimizer = _optimizer_attributes.convertOptimizerToDict(
            optimizer, self._attribute_tracker, self._options,
            self.isCompiled())

        if dict_optimizer != self._dict_optimizer:
            self._dict_optimizer = dict_optimizer
            self._ipu_optimizer_is_dirty = True

        # If we need and can update the optimizer now: do it.
        if self.isAttachedToDevice():
            self._update_optimizer_if_needed()
            self._write_optim_state_dict_if_needed()

    def _get_module_and_name(self, n):
        """
        Given a nested attribute path, return `(module, name)` such that
        `module` is the object which contains the attribute `name`, relative to
        `self._model`.

        This makes it easy to access nested attributes with
        `getattr` and `setattr`, using the argument splat `*a` operator, i.e.:

        ```
        getattr(*self._get_module_and_name("some_module.layer_one.weight"))
        ```

        gets the attribute `self._model.some_module.layer_one.weight`.
        """
        m = self._model
        name = n
        sn = n.rpartition(".")
        if sn[1] == ".":
            m = m.get_submodule(sn[0])
            name = sn[2]
        return m, name

    @_impl.destroyDispatcherOnExit
    def _compileWithDispatch(self, in_tensors, executable_filename=None):
        with _OverwriteContextManager():
            module_namescope = None
            if self.options._module_namescope_enabled:  # pylint: disable=protected-access
                module_namescope = _impl.NameScopeHook(self._model)

            tensor_args = flattenTensorStructure(
                (in_tensors.args, in_tensors.kwargs))
            mlir_compiler_options = poptorch_core.CompilerOptions()
            mlir_compiler_options.source_location_excludes = self._options._source_location_excludes  # pylint: disable=line-too-long, protected-access

            dispatch_failed = False
            try:  # pylint: disable=too-many-nested-blocks
                # Create the graph. Future captured calls will be written into this
                # graph behind the scenes.
                poptorch_core.createGraph(
                    poptorch_core.TracingMode(
                        poptorch_core.TracingMode.PopART), tensor_args,
                    mlir_compiler_options)

                # Move the model parameters to the ipu and take a copy to load the
                # originals back once this has finished
                cpu_params = dict(self._model.named_parameters())
                cpu_buffers = dict(self._model.named_buffers())
                cpu_state = self._model.state_dict(keep_vars=True)
                # We need to remove the PoptorchBuffer and PoptorchParam annotations
                # before compiling the model. In addition, we must unwrap the whole
                # model to prevent IPU to CPU copies when accessing the state_dict.
                _impl.unwrapModelIfNecessary(self._model)
                if self.per_replica_params is not None:
                    for name, param in cpu_params.items():
                        if name in self.per_replica_params:
                            if param.shape == torch.Size([]):
                                raise _impl.createPoptorchError(
                                    "Scalars cannot be passed as per-replica "
                                    "weight tensor values")
                            param_tensor = param.narrow(0, 0, 1).squeeze(dim=0)
                            setattr(*self._get_module_and_name(name),
                                    torch.nn.Parameter(param_tensor))
                d = torch.device("ipu:0")
                poptorch_core.startParametersMove()
                self._model.to(d)
                poptorch_core.endParametersMove()

                # If there were any parameters and buffers (tensors), which were
                # aliases on the CPU (shared the same Python ID), these will have
                # become separate IPU tensors during the copy to IPU
                #
                # Find all such tensors, and then
                # 1. Keep a map from them to the earliest cpu tensor in the
                #    cpu_state dict.
                # 2. Replace IPU tensors which are not but should be aliases with
                #    that matching the earliest.
                # NB the "original" name is based on order of addition of the
                # tensors/modules and may not be a name of the parmeter which
                # replaced another, e.g. the case of "weight tying", but the
                # name of the "replaced". However, no names will be lost but the
                # aliases simply harmonised to be matching tensors on CPU and IPU.
                state = self._model.state_dict(keep_vars=True)
                tensors = collections.defaultdict(list)
                for name, tensor in cpu_state.items():
                    tensors[id(tensor)].append(name)
                # A map of parameters and buffers (tensors) on the CPU which share
                # the same python id, to the earliest tensor.
                cpu_aliases = {}

                aliases = [v for v in tensors.values() if len(v) > 1]
                for a in aliases:
                    # NB original matches that in model.named_x() as both this as
                    # model.state_dict() loop he same  OrderedDicts in same order
                    # and the named versions return only the first instances
                    original = a[0]

                    for other in a[1:]:
                        setattr(*self._get_module_and_name(other),
                                state[original])
                        cpu_aliases[other] = original

                # Map named unique parameters and buffers on the IPU.
                params = dict(self._model.named_parameters())

                poptorch_core.mapParamsToNames(tuple(params.keys()),
                                               tuple(params.values()))

                buffers = dict(self._model.named_buffers())

                poptorch_core.mapParamsToNames(tuple(buffers.keys()),
                                               tuple(buffers.values()))

                old_addresses = _impl.getBufferAndParameterAddresses(
                    self._model)

                if self.per_replica_params is not None:
                    for name, param in cpu_params.items():
                        if name in self.per_replica_params:
                            poptorch_core.setPerReplica(
                                name, param, *self.per_replica_params[name])

                poptorch_core.startDispatch()
                _impl.setDispatchTracing(True)
                _impl.setIpuContext(True)

                for _, hook in PoplarExecutor._precompile_hooks.items():
                    hook()

                self._options._execution_strategy.onStartTracing()  # pylint: disable=protected-access

                # The optimizer was created using the CPU model, therefore it points
                # at CPU tensors.  We need to remap those to IPU tensors.
                # We just moved '_model' to the IPU, therefore we need to join the
                # two maps and then remap the parameters from the optimizer.
                # From:
                #
                # cpu_tensors[name] = cpu_data_ptr
                # ipu_tensors[name] = ipu_tensor
                #
                # we build:
                #
                # cpu_to_ipu[cpu_data_ptr] = ipu_tensor
                #
                # And then remap all the tensors from group["params"]
                if self._training:
                    cpu_tensors = {
                        **cpu_buffers,
                        **cpu_params,
                    }
                    ipu_tensors = _impl.getBufferAndParameterTensors(
                        self._model)
                    cpu_to_ipu = {
                        cpu_tensors[n].data_ptr(): ipu
                        for n, ipu in ipu_tensors.items()
                    }
                    for index, group in enumerate(
                            self._optimizer.param_groups):
                        torch.ops.poptorch.optimizer_group(
                            index, [
                                cpu_to_ipu[cpu.data_ptr()]
                                for cpu in group["params"]
                            ])

                for idx, t in enumerate(tensor_args):
                    if t.requires_grad:
                        raise _impl.createPoptorchError(
                            "An input tensor to an IPU model can not have "
                            f"requires_grad set to True, however input {idx} "
                            f"does: {t}\nYou can set requires_grad=True from "
                            "within the model as an alternative, and return "
                            "gradients as outputs to your model, if required.")

                d = torch.device("ipu:0")
                # Move all the inputs to the IPU
                tensor_args = [t.to(d) for t in tensor_args]
                # Re-inject moved tensors in args and kwargs:
                args, kwargs = reconstructTensorStructure(
                    (in_tensors.args, in_tensors.kwargs), tensor_args)

                result = self._model(*args, **kwargs)
                if result is not None:
                    self._outputs_structure = result
                    output = flattenTensorStructure(result)

                    for x in output:
                        if not isOnIpu(x):
                            warnings.warn(
                                "Output expected to be on the IPU but is on %s"
                                % x.device.type)

                    output = [
                        out.int()
                        if out.dtype == torch.long and isOnIpu(out) else out
                        for out in output
                    ]
                    output = [
                        out.float()
                        if out.dtype == torch.double and isOnIpu(out) else out
                        for out in output
                    ]
                    poptorch_core.startOutputsMove()
                    output = [out.cpu() for out in output]
                    poptorch_core.endOutputsMove()

                poptorch_core.finalizeGraph()
            except:
                dispatch_failed = True
                raise
            finally:
                self._options._execution_strategy.onEndTracing()  # pylint: disable=protected-access

                for _, hook in PoplarExecutor._postcompile_hooks.items():
                    hook()

                _impl.setIpuContext(False)
                _impl.setDispatchTracing(False)
                # Turn off the dispatcher.
                poptorch_core.endDispatch(dispatch_failed)

                # Reload the cpu model state
                # Get the buffer and parameter addresses after the model has ran
                # but before resetting the model back to the cpu
                new_addresses = _impl.getBufferAndParameterAddresses(
                    self._model)

                def _set_param(k, v):
                    setattr(*self._get_module_and_name(k), cpu_params[v])

                for k in cpu_params:
                    cpu_params[k].__class__ = torch.nn.Parameter
                    _set_param(k, k)

                # Restore aliased parameters/buffers which will not be represented
                # in cpu_params or cpu_buffers
                for k, v in cpu_aliases.items():
                    _set_param(k, v)

                for k in cpu_buffers:
                    setattr(*self._get_module_and_name(k), cpu_buffers[k])

                # Re-install the Poptorch annotations for buffers and parameters
                _impl.rewrapModelIfNecessary(self._model)

                # Check that the buffer and parameter addresses haven't been changed
                # in the model
                # Note: this is done after resetting the model back to the cpu so
                # that errors thrown by this don't stop the model being in a valid
                # state
                _impl.errorOnBufferOrParameterAddressChanges(
                    old_addresses, new_addresses)

                if module_namescope is not None:
                    module_namescope.remove()

            # We only reach this point if dispatch didn't fail
            if executable_filename is not None:
                # Compile the captured graph using PopART.
                executable = poptorch_core.processDispatchAndImportExecutable(
                    self._options.toDict(), accessAttributes, self._training,
                    self._dict_optimizer,
                    list(self._options.anchored_tensors.values()),
                    executable_filename)
            else:
                # Compile the captured graph using PopART.
                executable = poptorch_core.compileWithManualTracing(
                    self._options.toDict(), accessAttributes, self._training,
                    self._dict_optimizer,
                    list(self._options.anchored_tensors.values()))

        return executable

    @_impl.traceMethod("modelCompilation")
    def _compile(self, in_tensors):
        """On POD we want to separate compilation from device
        initialisation because we want only one process to compile the model,
        but ``loadEngineAndConnectStreams()`` must happen at the same time in
        all the processes (Because they need to talk to each other during the
        initialisation process).

        This is achieved by calling the equivalent of ``compileAndExport()``
        from one of the processes: this will populate the PopART cache with
        the executable. (We use a temp file because we don't need the result,
        we just want the executable to be added to the cache).

        The caller will then call the regular ``_compile()`` method in all the
        processes at the same time and they should all hit the cache.
        """
        # Compile the poplar executable based on the batchsize.
        in_tensors_trace_view = self._preprocessGraph(in_tensors)

        # Note: in single process execution or if the cache is disabled
        # should_compile will always be False.
        with _impl.distributedCacheLock(self._model,
                                        self._options) as should_compile:
            # Only the first process should compile
            if should_compile:
                self._executable = self._compileWithDispatch(
                    in_tensors_trace_view)

        # In distributed execution mode:
        # At that point only the first process will have a compiled executable:
        # trigger the compilation process in all the other processes.
        if not self.isCompiled():
            self._executable = self._compileWithDispatch(in_tensors_trace_view)

        # Load the engine and connect the streams in all the processes.
        #
        # Note: no sync point was added because we expect the above
        # compileWithDispatch call to be quick as all the processes should
        # hit the cache.
        #
        # If the cache is disabled then we expect the compilation process
        # to roughly take the same amount of time in all processes.
        #
        # Note: if multiple processes run on the same host, it's recommended
        # to enable executable caching to avoid out of memory issues due
        # to concurrent compilation processes.
        if self._options.connection_type != enums.ConnectionType.Never:
            poptorch_core.loadEngineAndConnectStreams(self._executable)

        self._is_attached = self.isAttachedToDevice()

        # PopTorch might have attached to a device either during
        # compileWithDispatch (if connection type is set to Always) or
        # during loadEngineAndConnectStreams (if OnDemand is used),
        # either way this will have occurred in the C++ backend, *not* using
        # PoplarExecutor.attachToDevice(), therefore we need to manually
        # call the _on_device_attach() trigger here.
        if self._is_attached:
            self._on_device_attach()

    @_impl.traceMethod("graphPreprocessing")
    def _preprocessGraph(self, in_tensors):
        self._executable_inputs = in_tensors.clone()
        in_tensors_trace_view = in_tensors.clone()

        def remove_requires_grad(tensor):
            if not isinstance(tensor, torch.Tensor):
                return tensor
            if tensor.requires_grad:
                tensor = tensor.detach()
                logger.warning(
                    "Input tensor has requires_grad=True set. "
                    "This tensor will be detached because backward pass via "
                    "inputs is not supported.")
            return tensor

        in_tensors_trace_view.forEach(self._narrow_tensor)
        in_tensors_trace_view.forEach(remove_requires_grad)
        return in_tensors_trace_view

    def compile(self, *args, **kwargs) -> None:
        """Takes the same arguments as the wrapped PyTorch `model.__call__`.

        Trace and compile the wrapped model if no executable has been
        created yet.

        Note: The executable created by this method can only be executed,
        it cannot be exported to file.
        To precompile and save to file use
        :py:meth:`~poptorch.PoplarExecutor.compileAndExport`
        """
        in_tensors = self._args_parser(args, kwargs, False)
        if self._executable is not None:
            logger.warning(
                "Call to compile() ignored: the executable is already compiled"
            )
        else:
            self._compile(in_tensors)

    @_impl.traceMethod("loadExecutable")
    def loadExecutable(self, filename: str) -> None:
        """Load an executable previously generated using
        :py:meth:`~poptorch.PoplarExecutor.compileAndExport`
        """
        serialized_data = poptorch_core.importPoptorchMetadataFromFile(
            filename)

        try:
            data = _poptorch_data.parse(serialized_data,
                                        self._poptorch_version)
        except AssertionError as e:
            raise _impl.createPoptorchError("Invalid file %s: %s" %
                                            (filename, e))

        in_tensors_trace_view = self._preprocessGraph(data.executable_inputs)
        self._executable = self._compileWithDispatch(
            in_tensors_trace_view, executable_filename=filename)

        self._is_attached = self.isAttachedToDevice()

        if self._is_attached:
            self._on_device_attach()

    def save(self,
             filename: str,
             export_model: bool = True,
             save_rng_state: bool = True):
        """Save the compiled model to file.

        :param filename: Where to save the compiled executable.
        :param export_model: If `True` the Torch model will be saved in
            the file alongside the executable. :py:func:`~poptorch.load` can
            be used to restore both the original Torch model, the PopTorch
            model and the executable.
            If `False` then only the executable will be exported and it will
            be the user's responsibility to call
            :py:func:`~poptorch.inferenceModel` or
            :py:func:`~poptorch.trainingModel` to re-create the PopTorch model
            before calling :py:meth:`~poptorch.PoplarExecutor.loadExecutable`
            to restore the executable.
        :param save_rng_state: If `True` the random number generator's state
            and seed will be saved in the file alongside the executable.
        """
        if not self.isCompiled():
            raise _impl.createPoptorchError(NO_EXECUTABLE_ERR)
        dst_dir = os.path.dirname(filename)
        if dst_dir:
            if os.path.exists(dst_dir):
                assert os.path.isdir(dst_dir), ("Destination folder {dst_dir} "
                                                "is not a directory")
            else:
                os.makedirs(dst_dir)
        if os.path.isdir(filename):
            dirname = filename
            filename = os.path.join(dirname, "model.poptorch")
            logger.warning("save(): %s is a directory, saving model to %s",
                           dirname, filename)

        data = _poptorch_data.PoptorchData(self._poptorch_version,
                                           self._executable_inputs,
                                           self._options)
        if export_model:
            data.training = self._training
            data.model = self.model
            data.optimizer = self._optimizer

        if save_rng_state:
            data.rng_state = self.rng_state

        serialized_data = pickle.dumps(data, protocol=4)

        with self._profiling.tracepoint("saveExecutableToFile"):
            poptorch_core.saveExecutableToFile(self._executable, filename)
            poptorch_core.appendPoptorchMetadataToFile(serialized_data,
                                                       filename)

    @property
    def rng_state(self) -> List[int]:
        """Return the random number generator's seed & state of
        the compiled model."""
        if not self.isCompiled():
            raise _impl.createPoptorchError(NO_EXECUTABLE_ERR)
        if self._host_rng_state_is_dirty:
            self._host_rng_state_is_dirty = False
            self._cached_rng_state = [
                poptorch_core.getRandomSeed(self._executable)
            ] + poptorch_core.getRngState(self._executable)
        return self._cached_rng_state

    @rng_state.setter
    def rng_state(self, state: List[int]):
        """Set the random number generator's seed & state for the compiled
        model."""
        if not self.isCompiled():
            raise _impl.createPoptorchError(NO_EXECUTABLE_ERR)
        self._host_rng_state_is_dirty = False
        self._cached_rng_state = state.copy()
        if self.isAttachedToDevice():
            self._copyRngStateToDevice()

    def _copyRngStateToDevice(self):
        poptorch_core.setRngState(self._executable, self._cached_rng_state[0],
                                  self._cached_rng_state[1:])

    @_impl.traceMethod("compileAndExport")
    def compileAndExport(self,
                         filename: str,
                         *args: List['torch.Tensor'],
                         export_model: bool = True,
                         **kwargs: Dict[str, 'torch.Tensor']):
        """Precompile an executable and save it to file.

        ``args`` and ``kwargs`` are the same arguments as the wrapped PyTorch
        ``model.__call__``

        :param filename: Where to save the compiled executable.
        :param export_model: If `True` the Torch model will be saved in
            the file alongside the executable. :py:func:`~poptorch.load` can
            be used to restore both the original Torch model, the PopTorch
            model and the executable.
            If `False` then only the executable will be exported and it will
            be the user's responsibility to call
            :py:func:`~poptorch.inferenceModel` or
            :py:func:`~poptorch.trainingModel` to re-create the PopTorch model
            before calling :py:meth:`~poptorch.PoplarExecutor.loadExecutable`
            to restore the executable.
        """

        self.compile(*args, **kwargs)
        self.save(filename, export_model)

    def cycleCount(self) -> int:
        """ Returns number of cycles which the IPU ran.

            You must run the model on IPU hardware before calling this method.

            :returns: number of cycles on the IPU for the last modern run. If
              you are using replicas, the returned value represents the first
              number of cycles for the first replica only."""

        # pylint: disable=protected-access
        popart_options = self._options._Popart
        if not popart_options.options['instrumentWithHardwareCycleCounter']:
            err_msg = ("Cycle count logging is disabled. Please set option "
                       "logCycleCount to True to enable.")
            raise _impl.createPoptorchError(err_msg)

        if not self.isCompiled():
            err_msg = ("Please run the model at least once before obtaining "
                       "cycle count.")
            raise _impl.createPoptorchError(err_msg)

        return poptorch_core.cycleCount(self._executable)

    def __call__(self, *args: List['torch.Tensor'],
                 **kwargs: Dict[str, 'torch.Tensor']):
        """
        Takes the same arguments as the wrapped PyTorch `model.__call__`.

        .. note:: The first time the :py:class:`~poptorch.PoplarExecutor`
            wrapper is called, the wrapped model will be traced and compiled.

        """
        assert self._options.connection_type != enums.ConnectionType.Never, (
            "Trying to run a model on an offline device "
            "(ConnectionType.Never): use model.compile(inputs) instead of"
            " model(inputs)")

        # If it is compiled we take the fast path, if not we convert lists to tuples.
        in_tensors = self._args_parser(args, kwargs, self.isCompiled())

        if not self.isCompiled():
            self._compile(in_tensors)

        if not self._is_attached:
            self.attachToDevice()
        if not self._training:
            # If this is an inference model: check if the same model is not being
            # trained on a different IPU.
            # If it is: make sure the weights are updated.

            copyWeightsToHostIfNeeded = getattr(self._user_model,
                                                "copyWeightsToHostIfNeeded",
                                                None)
            if callable(copyWeightsToHostIfNeeded):
                copyWeightsToHostIfNeeded()
                if self._host_weights_version != \
                        self._user_model._host_weights_version:
                    # Weights have now been updated on the Host: copy them to
                    # the second IPU.
                    logger.debug("Implicit copyWeightsToDevice()")
                    self.copyWeightsToDevice()
                    self._host_weights_version = \
                            self._user_model._host_weights_version

        self._executable_inputs.validateInputs(in_tensors)
        in_tensors_flat = in_tensors.asPackedFlatTuple(self._executable_inputs)

        # Update the optimizer state on the IPU if needed.
        self._write_optim_state_dict_if_needed()
        # Execute the poplar executable with the full size (batch * device interations)
        with self._profiling.tracepoint("modelExecution"):
            output = poptorch_core.execute(self._executable, in_tensors_flat)

        # Any anchored tensors will be returned at the end of the list
        # Pop them out and populate the anchor memory
        long_names = list(self._options.anchored_tensors.values())
        for long_name in reversed(long_names):
            tensor = output.pop()
            keys = [
                key for key, value in self._options.anchored_tensors.items()
                if value == long_name
            ]
            for key in keys:
                self._anchor_memory[key] = tensor

        self._host_rng_state_is_dirty = True
        if self._training:
            self._dirty_host_weights = True

        if self._optimizer and isinstance(self._optimizer, Optimizer):
            # The optimizer has been used on the IPU: its state on the host
            # is now out of date.
            self._optimizer.host_state_is_dirty = True

        # Provide a useful error message if the user attempts to call
        # backward() on an output tensor
        self._assign_backward_error(output)

        if self._outputs_structure is not None:
            # Only return the IPU tensors
            return reconstructTensorStructure(self._outputs_structure, output,
                                              isOnIpu)
        if len(output) == 0:
            return None
        if len(output) > 1:
            return output
        return output[0]

    def _assign_backward_error(self, input):
        def error_on_backward():
            raise _impl.createPoptorchError(
                "backward() cannot be called explicitly on "
                "outputs of a PopTorch model. If you're using a trainingModel, "
                "the backwards pass is performed automatically when invoking "
                "the model. If you're using an inferenceModel, you should use "
                "a trainingModel instead.")

        if isinstance(input, (list, tuple)):
            for element in input:
                self._assign_backward_error(element)
        elif isinstance(input, torch.Tensor):
            input.backward = error_on_backward

    def getPerfCounters(self):
        """Return performance counters for the last execution of the model.

        Return the values (in fractional seconds) of the performance counters
        corresponding to the latest run of the model. The reference point of
        the returned value is undefined, however the difference between values
        is valid.

        The returned object is a dictionary where they keys correspond to each
        of the following events:
        * 'input': the IPU requesting an input tensor
        * 'input_complete': an input tensor having been transferred
        * 'output': the IPU requesting to transmit an output tensor
        * 'output_complete': an output tensor having been transferred

        The values of the dictionary are nested lists. The first level of
        nesting corresponds to an input or output index. The second level list
        contains the actual values as fractional seconds.

        Examples:
        * dict['input'][1][3]: performance counter for the second input
        tensor being requested on the third iteration of the model
        * dict['output_complete'][0][0]: performance counter the first
        output tensor having been transferred on the first iteration of
        the model
        """
        if not self.isCompiled():
            return {
                'input': [[]],
                'input_complete': [[]],
                'output': [[]],
                'output_complete': [[]]
            }

        def normalize(timestamps):
            if len(timestamps) == 0:
                return [[]]
            return timestamps

        values = poptorch_core.getTimestamps(self._executable)
        return {
            'input': normalize(values[0]),
            'input_complete': normalize(values[1]),
            'output': normalize(values[2]),
            'output_complete': normalize(values[3])
        }

    def _computeLatency(self, from_event: str,
                        from_reduce: Callable[[List[float]], float],
                        to_event: str,
                        to_reduce: Callable[[List[float]], float]):
        """Computes latency figures between two performance counters.

        :param from_event: Key for starting performance counter.
        :param from_reduce: Reduction function for starting counters.
        :param to_event: Key for ending performance counter.
        :param to_reduce: Reduction function for ending counters.

        .. seealso:: :py:meth:`~poptorch.PoplarExecutor.getPerfCounters` for
            the list of keys allowed.
        """
        perf_counters = self.getPerfCounters()
        start_times = []
        end_times = []
        durations = []

        num_inputs = len(perf_counters[from_event])
        for step in range(0, len(perf_counters[from_event][0])):
            start_times.append(
                from_reduce([
                    perf_counters[from_event][i][step]
                    for i in range(0, num_inputs)
                ]))

        num_outputs = len(perf_counters[to_event])
        for step in range(0, len(perf_counters[to_event][0])):
            end_times.append(
                to_reduce([
                    perf_counters[to_event][i][step]
                    for i in range(0, num_outputs)
                ]))

        if len(end_times) == 0:
            return (0., 0., 0.)

        # It is possible to have more input timestamps than output timestamps
        # due to other options such as gradient accumulation and output modes.
        # Whatever the case, the number of input ticks will always be divisible
        # by the number of output ticks.
        assert len(start_times) % len(end_times) == 0, \
            "Internal PopTorch error: mismatching number of start timestamps" \
            " and ending timestamps when calculating latency"

        # Find the group of input ticks corresponding to each output tick and
        # replace the whole set by its minimum
        factor = len(start_times) // len(end_times)
        start_groups = [
            min(start_times[i:i + factor])
            for i in range(0, len(start_times), factor)
        ]

        durations = list(
            map(lambda v: v[1] - v[0], zip(start_groups, end_times)))

        avg = sum(durations) / len(durations)
        return (min(durations), max(durations), avg)

    def getHostIpuLatency(self):
        """Return Host-IPU latency for the last execution of the model.

        The Host-IPU latency is the interval of time (in fractional seconds)
        between the first input tensor being requested and the last input
        tensor being transferred to the IPU.

        The result is a tuple containing the minimum, maximum and average
        latency for the iterations corresponding to the latest invocation of
        the model.
        """
        return self._computeLatency('input', min, 'input_complete', max)

    def getComputeLatency(self):
        """Return compute latency for the last execution of the model.

        The compute latency is the interval of time (in fractional seconds)
        between the last input tensor being transferred to the IPU and the
        last output tensor becoming available.

        The result is a tuple containing the minimum, maximum and average
        latency for the iterations corresponding to the latest invocation of
        the model.
        """
        return self._computeLatency('input_complete', max, 'output', max)

    def getIpuHostLatency(self):
        """Return IPU-Host latency for the last execution of the model.

        The IPU-Host latency is the interval of time (in fractional seconds)
        between the first output tensor becoming available and the last output
        tensor being written back to the host.

        The result is a tuple containing the minimum, maximum and average
        latency for the iterations corresponding to the latest invocation of
        the model.
        """
        return self._computeLatency('output', min, 'output_complete', max)

    def getLatency(self):
        """Return round-trip latency for the last execution of the model.

        The round-trip latency is the interval of time (in fractional seconds)
        between the first input tensor being requested and the last output
        tensor being written back to the host.

        The result is a tuple containing the minimum, maximum and average
        latency for the iterations corresponding to the latest invocation of
        the model.
        """
        return self._computeLatency('input', min, 'output_complete', max)

    def destroy(self) -> None:
        """Destroy the model: release the IPUs and the executable.
        """
        if not self.isCompiled():
            return
        if self._training:
            self.copyWeightsToHostIfNeeded()
            # Sync the optimizer's state dict back to host
            self._optimizer.state_dict()

        del self._executable
        self._executable = None

        if not self._training:
            return

        # unwrap the model, parameters and buffers
        if not _impl.isWrapped(self._user_model):
            raise _impl.createPoptorchError("model was never wrapped")
        _impl.unwrapModelIfNecessary(self._user_model)

    def _narrow_tensor(self, tensor):
        """There are two concepts of batch size. First is the "model" batch
        size then there is the concept of batching at the popart level.
        Here we divide by the popart batch size so the trace "sees" the
        model batch size but when we call execute we pass the full batch
        and popart will partition it up."""

        input_group_count = self._options.replication_factor // \
                            self._options.input_group_size
        # Input will be in form of [ModelBatchSize * BatchPerStep, ...] so we
        # should slice it up so we compile by the ModelBatchSize alone.
        extra_poplar_batch_dims = self._options.device_iterations * \
            input_group_count * self._options.Training.gradient_accumulation

        if not isinstance(tensor, torch.Tensor):
            return tensor

        b_size = 1 if not tensor.size() else tensor.size()[0]
        assert b_size % extra_poplar_batch_dims == 0, (
            "Invalid batch dimension: In the input %s, the batch "
            "dimension (%d) must be a multiple of "
            "Options.deviceIterations(%d) * "
            "(Options.replicationFactor(%d) / "
            "Options.inputReplicaGrouping.input_group_size(%d)) * "
            "Options.Training.gradientAccumulation(%d) = %d "
            "because it is used to calculate the batch size which will "
            "be executed on the device in any given iteration. For a "
            "full explanation see the batching semantics page of the "
            "documentation."
        ) % (tensor.shape, b_size, self._options.device_iterations,
             self._options.replication_factor, self._options.input_group_size,
             self._options.Training.gradient_accumulation,
             extra_poplar_batch_dims)
        return tensor if tensor.shape == torch.Size([]) else tensor.narrow(
            0, 0, b_size // extra_poplar_batch_dims)

    def isAttachedToDevice(self) -> bool:
        """Returns true, if the target device has been attached. False,
        otherwise.
        """
        if not self.isCompiled():
            return False

        return poptorch_core.isAttachedToDevice(self._executable)

    def isCompiled(self) -> bool:
        """Returns true if the model has been compiled (and not destroyed).
        False, otherwise."""
        return bool(self._executable)

    def detachFromDevice(self) -> None:
        """Detach from target device. Before calling this function, the device
        must be attached (and the model compiled)."""
        if not self.isCompiled():
            raise _impl.createPoptorchError(NO_EXECUTABLE_ERR)

        if not self._is_attached:
            raise _impl.createPoptorchError("Device is not attached")

        # Read all the states back before detaching
        _ = self.rng_state
        if self._training:
            self.copyWeightsToHostIfNeeded()
            self._read_optim_state_dict_if_needed()

        poptorch_core.detachFromDevice(self._executable)
        self._is_attached = False

    def attachToDevice(self) -> None:
        """Attach to target device. Before calling this function, the device
        must be detached and the model compiled."""
        if not self.isCompiled():
            raise _impl.createPoptorchError(NO_EXECUTABLE_ERR)
        assert self._options.connection_type != enums.ConnectionType.Never, (
            "Trying to attach to an offline device"
            " (ConnectionType.Never)")

        if self._is_attached:
            raise _impl.createPoptorchError("Device is already attached")

        poptorch_core.attachToDevice(self._executable)
        poptorch_core.loadEngineAndConnectStreams(self._executable)
        self._is_attached = True
        self._on_device_attach()


def _registerHook(hooks, new_hook) -> torch.utils.hooks.RemovableHandle:
    handle = torch.utils.hooks.RemovableHandle(hooks)
    hooks[handle.id] = new_hook
    return handle


def registerPreCompileHook(hook: Callable
                           ) -> torch.utils.hooks.RemovableHandle:
    """Register a hook that is called before model compilation.

    Raises a ``RuntimeError` if the hook is not callable.

    :param hook: A callable that is ran before model compilation begins.
    :returns: a :py:class:`torch.utils.hooks.RemovableHandle` that can be used
        to remove the hook using :py:func:`~remove`
    """
    if not callable(hook):
        raise RuntimeError("Pre-compile hook must be callable")
    hooks = PoplarExecutor._precompile_hooks  # pylint: disable=protected-access
    return _registerHook(hooks, hook)


def registerPostCompileHook(hook: Callable
                            ) -> torch.utils.hooks.RemovableHandle:
    """Register a hook that is called after model compilation.

    Raises a ``RuntimeError` if the hook is not callable.

    :param hook: A callable that is ran after model compilation ends.
    :returns: a :py:class:`torch.utils.hooks.RemovableHandle` that can be used
        to remove the hook using :py:func:`~remove`
    """
    if not callable(hook):
        raise RuntimeError("Post-compile hook must be callable")
    hooks = PoplarExecutor._postcompile_hooks  # pylint: disable=protected-access
    return _registerHook(hooks, hook)
