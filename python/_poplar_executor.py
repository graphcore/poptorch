# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import collections
import copy
import itertools
import json
import os
import pickle
import re
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
from . import optim
from . import profiling
from . import poptorch_core  # type: ignore
from . import _poptorch_data
from ._utils import (accessAttributes, unrollTensorList,
                     reconstruct_output_structure)
from ._logging import logger
from .experimental import IPUScope
from .options import Options, PipelinedExecution, ShardedExecution
from .optim import Optimizer

NO_EXECUTABLE_ERR = "Model has not been compiled or has been destroyed."

# Some modules will still work even if the buffer address changes during tracing
BUFFERS_CAN_CHANGE = (
    torch.nn.BatchNorm1d,
    torch.nn.modules.batchnorm.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.modules.batchnorm.BatchNorm3d,
)


# pylint: disable=too-many-public-methods
class PoplarExecutor:
    """ This class should not be created directly but is a wrapper around
    the model that was passed into `inferenceModel` or `trainingModel`.
    It only has a few methods which can be used to interface with the IPU.
    """

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

        options.Precision.autocast_policy.apply(self._user_model, options)

        if training:
            # TODO(T51159): Add support for dispatch tracing + training.
            assert options.Jit.trace_model, "training not supported for now"
            self._attribute_tracker = \
                    _optimizer_attributes.OptimizerAttrTracker(
                        options)
            if options.defaultOutputMode():
                # In training it makes sense to see only the last result, by default.
                options.outputMode(enums.OutputMode.Final)
            if not optimizer:
                optimizer = optim.SGD(self._user_model.parameters(), lr=0.01)
            model = _impl.OptimizerWrapper(model, optimizer)
        else:
            if options.defaultOutputMode():
                # In inference it makes sense to see all the results, by default.
                options.outputMode(enums.OutputMode.All)

            if options.Training.gradient_accumulation != 1:
                err_msg = (
                    "You must set " +
                    "poptorch.Options().Training.gradientAccumulation to 1 " +
                    "or leave it as its default value (1) when running a " +
                    "poptorch.inferenceModel().")

                is_pipelined = (isinstance(options._execution_strategy,
                                           PipelinedExecution)
                                and not isinstance(options._execution_strategy,
                                                   ShardedExecution))
                if is_pipelined:
                    err_msg += (" Use poptorch.Options().deviceIterations " +
                                "to process a sufficient number of batches " +
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
                    if parent():
                        return parent().copyWeightsToHostIfNeeded()
                    return False

                def destroy(self):
                    """Destroy the model: release the IPUs and the executable.
                    """
                    if parent():
                        parent().destroy()

                def __getattribute__(self, name):
                    if name == "_host_weights_version":
                        if not parent():
                            return None
                        return parent()._host_weights_version
                    if name in ("_buffers", "_parameters", "forward"):
                        self.copyWeightsToHostIfNeeded()
                    return object.__getattribute__(self, name)

                def __getattr__(self, name):
                    attribute = super().__getattr__(name)
                    if isinstance(attribute, torch.nn.parameter.Parameter):
                        self.copyWeightsToHostIfNeeded()
                    return attribute

                def state_dict(self,
                               destination=None,
                               prefix="",
                               keep_vars=False):
                    """Return a shallow copy of the wrapped model's state dictionary.

                    Note: all the elements in the state dictionary are
                    unwrapped which means the state can be reloaded in an
                    environment where PopTorch is not installed.
                    """
                    out = collections.OrderedDict()
                    for k, v in super().state_dict(destination, prefix,
                                                   keep_vars).items():
                        # If the object is wrapped then the shallow copy will
                        # call _impl._pickleUnwrapObject and the new object will be in
                        # the wrapped registry.
                        v = copy.copy(v)
                        # Unwrap the object if needed.
                        out[k] = _impl.unwrapIfWrapped(v)
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
                    if parent():
                        parent().copyWeightsToHostIfNeeded()

                    return object.__getattribute__(self, name)

                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    if kwargs is None:
                        kwargs = {}
                    return super().__torch_function__(func, types, args,
                                                      kwargs)

            for p in self._user_model.parameters():
                p.__class__ = PoptorchParameter

            class PoptorchBuffer(torch.Tensor):
                def __getattribute__(self, name):
                    if parent():
                        parent().copyWeightsToHostIfNeeded()

                    return super().__getattribute__(name)

                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    if kwargs is None:
                        kwargs = {}
                    return super().__torch_function__(func, types, args,
                                                      kwargs)

            for b in self._user_model.buffers():
                if b.__class__ != torch.Tensor:
                    raise _impl.createPoptorchError(
                        "All buffers must be an instance of " + "torch.Tensor")
                b.__class__ = PoptorchBuffer

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
        return self._user_model.__repr__()

    def __getattr__(self, attr):
        return getattr(self._user_model, attr)

    @property
    def model(self) -> 'torch.nn.Module':
        """Access the wrapped Torch model."""
        return self._user_model

    @property
    def options(self) -> 'poptorch.Options':
        """Access to the options.

        .. seealso:: :py:class:`poptorch.Options`"""
        return self._options

    def _debugGetPopartIR(self) -> str:
        return poptorch_core._getPopartIR(self._executable)  # pylint: disable=protected-access

    def getTensorNames(self) -> List[str]:
        """Returns a list of all tensor names within the computational
        graph. Model must be compiled in advance.
        """

        assert self._executable is not None, "Model must be compiled " \
            "before calling getTensorNames"

        tensors = set()
        ir = json.loads(self._debugGetPopartIR())
        for op in ir.get('maingraph', {}):
            for t in op['inputs'] + op['outputs']:
                tensors.add(t['name'])

        return list(tensors)

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

    def _compileWithDispatch(self, in_tensors):
        def all_data(model):
            yield from model.named_parameters()
            yield from model.named_buffers()

        # Unpack the inputs.
        inputs = unrollTensorList(in_tensors.asTuple())

        # Store buffer and parameter memory addresses to make sure that these do
        # not change during dispatching (which would give wrong results in a Jit
        # trace)
        buff_param_addresses = self._buffer_parameter_addresses()

        with IPUScope(inputs,
                      parameters_and_buffers=all_data(self._model),
                      options=self._options) as ipu:
            outputs = self._model(*in_tensors.asTuple())
            ipu.outputs(outputs)
            self._outputs_structure = ipu._outputs_structure  # pylint: disable=protected-access
            self._error_on_buffer_parameter_address_change(
                buff_param_addresses)

        return ipu._executable  # pylint: disable=protected-access

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
        if self._options.Jit.trace_model:
            (in_tensors_trace_view,
             has_converted_any_half) = self._preprocessGraphTracing(in_tensors)

            trace_args = self._trace_model_and_get_compile_args(
                in_tensors, in_tensors_trace_view, has_converted_any_half)
        else:
            in_tensors_trace_view = self._preprocessGraphDispatcher(in_tensors)

        # Note: in single process execution or if the cache is disabled
        # should_compile will always be False.
        with _impl.distributedCacheLock(self._model,
                                        self._options) as should_compile:
            # Only the first process should compile
            if should_compile:
                if self._options.Jit.trace_model:
                    self._executable = poptorch_core.compileWithTrace(
                        *trace_args)
                else:
                    self._executable = self._compileWithDispatch(
                        in_tensors_trace_view)

        # In distributed execution mode:
        # At that point only the first process will have a compiled executable:
        # trigger the compilation process in all the other processes.
        if not self.isCompiled():
            if self._options.Jit.trace_model:
                self._executable = poptorch_core.compileWithTrace(*trace_args)
            else:
                self._executable = self._compileWithDispatch(
                    in_tensors_trace_view)

        # Load the engine and connect the streams in all the processes.
        #
        # Note: no sync point was added because we expect the above
        # compileWithTrace call to be quick as all the processes should
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
        # compileWithTrace (if connection type is set to Always) or
        # during loadEngineAndConnectStreams (if OnDemand is used),
        # either way this will have occurred in the C++ backend, *not* using
        # PoplarExecutor.attachToDevice(), therefore we need to manually
        # call the _on_device_attach() trigger here.
        if self._is_attached:
            self._on_device_attach()

    @_impl.traceMethod("graphPreprocessing")
    def _preprocessGraphTracing(self, in_tensors):
        in_tensors_trace_view = self._preprocessGraphCommon(in_tensors)
        # Normal bools don't get captured in python.
        has_converted_any_half = [False]

        def possiblyConvertFromHalf(tensor):
            if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.half:
                has_converted_any_half[0] = True
                return tensor.float()
            return tensor

        in_tensors_trace_view.forEach(possiblyConvertFromHalf)
        poptorch_core.processPrecisionOptions(self._options.Precision)
        return in_tensors_trace_view, has_converted_any_half

    @_impl.traceMethod("graphPreprocessing")
    def _preprocessGraphDispatcher(self, in_tensors):
        return self._preprocessGraphCommon(in_tensors)

    def _preprocessGraphCommon(self, in_tensors):
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

        in_tensors_trace_view, has_converted_any_half = \
                self._preprocessGraphTracing(
                    data.executable_inputs)

        if data.options.Jit.trace_model:
            trace_args = self._trace_model_and_get_compile_args(
                data.executable_inputs, in_tensors_trace_view,
                has_converted_any_half)
            self._executable = \
                    poptorch_core.processTraceAndImportExecutable(
                        *trace_args, filename)
        else:
            # TODO(T51159) Support dispatch tracing + serialized executables
            raise _impl.createPoptorchError("Not supported: can't deserialize "
                                            " dispatch traced executable.")
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
            the file alongside the executable. :py:func:`poptorch.load` can
            be used to restore both the original Torch model, the PopTorch
            model and the executable.
            If `False` then only the executable will be exported and it will
            be the user's responsibility to call
            :py:func:`poptorch.inferenceModel` or
            :py:func:`poptorch.trainingModel` to re-create the PopTorch model
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
            the file alongside the executable. :py:func:`poptorch.load` can
            be used to restore both the original Torch model, the PopTorch
            model and the executable.
            If `False` then only the executable will be exported and it will
            be the user's responsibility to call
            :py:func:`poptorch.inferenceModel` or
            :py:func:`poptorch.trainingModel` to re-create the PopTorch model
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
            err_msg = ("Cycle count logging is disabled. Please set option " +
                       "logCycleCount to True to enable.")
            raise _impl.createPoptorchError(err_msg)

        if not self.isCompiled():
            err_msg = ("Please run the model at least once before obtaining " +
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

        assert in_tensors.first_none == self._executable_inputs.first_none, (
            "Number of arguments mismatch: "
            f"{self._executable_inputs.first_none_arg} "
            f"arguments used to compile the model and "
            f"{in_tensors.first_none} provided this time")

        # Update the optimizer state on the IPU if needed.
        self._write_optim_state_dict_if_needed()
        # Execute the poplar executable with the full size (batch * device interations)
        with self._profiling.tracepoint("modelExecution"):
            output = poptorch_core.execute(self._executable,
                                           in_tensors.asTuple())

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
            return reconstruct_output_structure(self._outputs_structure,
                                                output)
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
        del self._executable
        self._executable = None

        if not self._training:
            return

        # Unwrap parmateres and buffers
        self.load_state_dict(self.state_dict())

        # unwrap the model
        if not _impl.isWrapped(self._user_model):
            raise _impl.createPoptorchError("model was never wrapped")

        _impl.unwrapIfWrapped(self._user_model)

    def _trace_with_warning_filter(self, in_tensors_trace_view_tuple):
        # Conditionally suppress the following jit warnings when the model
        # contains any non-deterministic nodes (e.g. dropout)
        rng_warnings = [
            "Trace had nondeterministic nodes",
            "the traced function does not match the corresponding output"
        ]

        def filterWarnings(warning):
            return not any([m in str(warning.message) for m in rng_warnings])

        warns = []
        with warnings.catch_warnings(record=True) as caught:
            try:
                traced = torch.jit.trace(self._model,
                                         in_tensors_trace_view_tuple)
            except RuntimeError as e:
                pattern = r'Type \'Tuple(\[.*\])\' cannot be traced'
                match = re.match(pattern, str(e))
                if match:
                    types = match.group(1)
                    raise TypeError(
                        "All forward function arguments used to compile and "
                        "run the model must be Tensors or (possibly nested) "
                        f"Lists and Tuples of Tensors (Got types: {types})."
                    ).with_traceback(e.__traceback__)
                raise e

            # pylint: disable=protected-access
            if poptorch_core.isGraphNondeterministic(traced._c):
                warns = list(filter(filterWarnings, caught))

        # Reissue remaining warnings
        for w in warns:
            warnings.warn_explicit(message=w.message,
                                   category=w.category,
                                   filename=w.filename,
                                   lineno=w.lineno)

        return traced

    def _getTraceNoOutput(self, in_tensors_trace_view_tuple):
        if not isinstance(self._model, torch.nn.Module):
            raise _impl.createPoptorchError(
                "Tracing a model returning no outputs is only " +
                "supported if the model is an instance of " +
                "torch.nn.Module or an instance of a subclass " +
                "of torch.nn.Module.")

        class AddFakeOutput(self._model.__class__):
            def forward(self, *args, **kwargs):
                super().forward(*args, **kwargs)
                return torch.tensor([0])

        old_class = self._model.__class__
        self._model.__class__ = AddFakeOutput
        traced = self._trace_with_warning_filter(in_tensors_trace_view_tuple)
        self._model.__class__ = old_class

        return traced

    def _buffer_parameter_addresses(self):
        # Obtains dictionaries of the data ptr addresses of every buffer
        # and parameter

        buffer_addresses = {}
        for module_name, module in self._model.named_modules():
            if isinstance(module, BUFFERS_CAN_CHANGE):
                continue

            for name, buff in module.named_buffers(prefix=module_name,
                                                   recurse=False):
                buffer_addresses[name] = buff.data_ptr()

        parameter_addresses = {}
        for name, param in self._model.named_parameters():
            parameter_addresses[name] = param.data_ptr()

        return buffer_addresses, parameter_addresses

    def _error_on_buffer_parameter_address_change(self, old_addresses):
        new_addresses = self._buffer_parameter_addresses()

        # Do the buffers first then paramters
        order = ["Buffer", "Parameter"]
        for idx, dic in enumerate(old_addresses):
            for name, address in dic.items():
                if name not in new_addresses[idx]:
                    err_msg = (order[idx] + " " + name + " is removed from " +
                               "the model when calling the forward method.")

                    raise _impl.createPoptorchError(err_msg)

                if address != new_addresses[idx][name]:
                    err_msg = (
                        order[idx] + " " + name + " is reassigned " +
                        "within the model when calling the forward " +
                        "method. This is not supported. Consider using self." +
                        name + ".copy_(src)" +
                        " to copy data from a source tensor, where src is " +
                        "the name of the source tensor.")
                    raise _impl.createPoptorchError(err_msg)

    @_impl.traceMethod("tracingModel")
    def _trace_model_and_get_compile_args(self, in_tensors,
                                          in_tensors_trace_view,
                                          has_converted_any_half):
        logger.info('Compiling the model using tracing')
        # CPU tracing doens't work for half types. We need to convert all half
        # layers to float, run tracing and revert the types to their original.
        half_layers = set()
        all_layers = list(self._model.named_modules())

        # Iterate in reverse to process inner layers first.
        for (name, layer) in reversed(all_layers):
            any_is_half = False
            for tensor in itertools.chain(layer.parameters(), layer.buffers()):
                if tensor.dtype == torch.half:
                    any_is_half = True
                    break

            if any_is_half:
                layer.float()
                half_layers.add(name)

        # From this point, if in_tensors_trace_view changes in value, the input
        # is modified in-place. To discover this, take a deep copy of the
        # inputs.
        in_tensors_trace_view_tuple = in_tensors_trace_view.asTuple()
        in_tensors_backup = None
        try:
            in_tensors_backup = copy.deepcopy(in_tensors_trace_view_tuple)
        # pylint: disable=bare-except
        except:
            # The trace should raise its own exception for invalid input types,
            # so simply keep in_tensors_backup as None. Note that a tensors with
            # requires_grad=True would fail here but such a tensor will have
            # been detached already.
            pass

        # We will trace using the normal trace view.
        # pylint: disable=protected-access
        self._options._execution_strategy.onStartTracing()

        # The CPU execution happening during trace represents the IPU codepath.
        _impl.setIpuContext(True)
        module_namescope = _impl.NameScopeHook(
            self._user_model
        ) if self.options._module_namescope_enabled else None

        # Override half so users can use it in their models.
        def internal_half(tensor):
            # Half is always outplace, if we don't clone, we might see the
            # casted tensor in places where the original tensor should be used
            # in the trace.
            tensor = tensor.clone()
            return _impl.internal_cast(tensor, torch.half)

        # Store the old half so it can be restored.
        old_half = torch.Tensor.half
        torch.Tensor.half = internal_half

        # Trace only a copy to avoid updating original weights during compilation.
        temp_model = copy.deepcopy(self._model.state_dict())

        added_dummy_output = False

        # Store buffer and parameter memory addresses to make sure that these do
        # not change during tracing (which would give wrong results in a Jit
        # trace)
        buff_param_addresses = self._buffer_parameter_addresses()

        try:
            self._trace = self._trace_with_warning_filter(
                in_tensors_trace_view_tuple)
        except RuntimeError as e:
            if "didn't return any values" in str(e):
                self._trace = self._getTraceNoOutput(
                    in_tensors_trace_view_tuple)
                added_dummy_output = True
            else:
                raise e

        self._error_on_buffer_parameter_address_change(buff_param_addresses)

        # Restore half to its old meaning.
        torch.Tensor.half = old_half

        # Revert the traced copy to the inital weights.
        self._trace.load_state_dict(temp_model, strict=False)
        self._model.load_state_dict(temp_model)

        # Restore to non-IPU codepath.
        if module_namescope:
            module_namescope.remove()
        _impl.setIpuContext(False)

        self._options._execution_strategy.onEndTracing()
        self._RestoreInputs(in_tensors_backup, in_tensors_trace_view_tuple)

        # Some of the trace layers of type float should be of type half.
        # The following works because the iterator is hierarchic,
        # yielding containers before contents.
        for name, layer in self._trace.named_modules():
            if name in half_layers:
                layer.half()

        # Convert back the original model as well.
        for name, layer in self._model.named_modules():
            if name in half_layers:
                layer.half()

        # We need to track the parameters from the traced model as this is what
        # the C++ graph sees.
        parameters = {
            **dict(self._trace.named_parameters()),
            **dict(self._trace.named_buffers())
        }

        # Track the original model parameters as well.
        model_parameters = {
            **dict(self._model.named_parameters()),
            **dict(self._model.named_buffers())
        }

        if has_converted_any_half[0]:
            # Get the originals back.
            in_tensors_as_half = in_tensors.clone()
            in_tensors_as_half.forEach(self._narrow_tensor)

            # Compile using the actual halves.
            return (self._trace._c, parameters,
                    in_tensors_as_half.asTuple(), has_converted_any_half[0],
                    self._options.toDict(), self._training,
                    self._dict_optimizer, accessAttributes, added_dummy_output,
                    list(self._options.anchored_tensors.values()),
                    model_parameters)
        return (self._trace._c, parameters,
                in_tensors_trace_view.asTuple(), has_converted_any_half[0],
                self._options.toDict(), self._training, self._dict_optimizer,
                accessAttributes, added_dummy_output,
                list(self._options.anchored_tensors.values()),
                model_parameters)

    def _narrow_tensor(self, tensor):
        """There are two concepts of batch size. First is the "model" batch
        size then there is the concept of batching at the popart level.
        Here we divide by the popart batch size so the trace "sees" the
        model batch size but when we call execute we pass the full batch
        and popart will partition it up."""

        # Input will be in form of [ModelBatchSize * BatchPerStep, ...] so we
        # should slice it up so we compile by the ModelBatchSize alone.
        extra_poplar_batch_dims = self._options.device_iterations * \
            self._options.replication_factor * \
            self._options.Training.gradient_accumulation

        if not isinstance(tensor, torch.Tensor):
            return tensor

        b_size = 1 if not tensor.size() else tensor.size()[0]
        assert b_size % extra_poplar_batch_dims == 0, (
            "Invalid batch dimension: In the input %s, the batch "
            "dimension (%d) must be a multiple of "
            "Options.deviceIterations(%d) * "
            "Options.replicationFactor(%d) * "
            "Options.Training.gradientAccumulation(%d) = %d "
            "because it is used to calculate the batch size which will "
            "be executed on the device in any given iteration. For a "
            "full explanation see the batching semantics page of the "
            "documentation.") % (tensor.shape, b_size,
                                 self._options.device_iterations,
                                 self._options.replication_factor,
                                 self._options.Training.gradient_accumulation,
                                 extra_poplar_batch_dims)
        return tensor.narrow(0, 0, b_size // extra_poplar_batch_dims)

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

    @classmethod
    def _RestoreInputs(cls, backup, post_trace):
        if isinstance(backup, torch.Tensor):
            assert isinstance(post_trace, torch.Tensor)
            post_trace.copy_(backup)
            return

        if isinstance(backup, (tuple, list)):
            assert isinstance(post_trace, (tuple, list))
            assert len(backup) == len(post_trace)

            for idx, backup_val in enumerate(backup):
                cls._RestoreInputs(backup_val, post_trace[idx])

            return

        # This implies that there is an input type or condition which does not
        # cause the tracer to fail, yet is none of the above types, or
        # alternatively, it is one of the above but the deepcopy failed.
        raise _impl.createPoptorchError("Unsupported input type or condition.")


def hasMlirSupportOnPlatform():
    return poptorch_core.mlirIsSupportedOnPlatform()
