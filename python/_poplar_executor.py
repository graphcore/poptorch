# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import copy
import copyreg
import ctypes
import json
import os
import pickle
import re
from typing import Callable, Dict, List, Optional
import weakref
import warnings
import torch

# Do not import any poptorch.* here: it will break the poptorch module
from . import _impl
from . import _args_parser
from . import _optimizer_attributes
from . import enums
from . import profiling
from . import poptorch_core  # type: ignore
from . import _poptorch_data
from ._logging import logger
from .ops import ATTR_PREFIX
from .options import Options


# Allow access to attributes
def accessAttributes(attribute_id_str):
    logger.debug("Accessing attributes with: %s", attribute_id_str)

    if not isinstance(attribute_id_str, (str)):
        raise ValueError("Wrong type for attribute_id_str")

    if not attribute_id_str.startswith(ATTR_PREFIX):
        raise ValueError("Invalid attribute_id_str")

    attribute_id = int(attribute_id_str[len(ATTR_PREFIX):], 16)

    # NB this is undefined behavior if attribute_id does not exist
    attributes = ctypes.cast(attribute_id, ctypes.py_object).value
    logger.debug(str(attributes))

    if attributes is None:
        return {}
    return attributes


NO_EXECUTABLE_ERR = "Model has not been compiled or has been destroyed."


class PoplarExecutor:
    """ This class should not be created directly but is a wrapper around
    the model that was passed into `inferenceModel` or `trainingModel`.
    It only has a few methods which can be used to interface with the IPU.
    """

    # pylint: disable=too-many-statements
    def __init__(self,
                 model: 'torch.nn.Module',
                 options: 'poptorch.Options',
                 training: bool,
                 poptorch_version: str,
                 optimizer: Optional['torch.optim.Optimizer'] = None,
                 user_model: Optional['torch.nn.Module'] = None):
        options = options or Options()
        self._user_model = user_model or model

        options.Precision.autocast_policy.apply(self._user_model, options)

        if training:
            self._attribute_tracker = \
                    _optimizer_attributes.OptimizerAttrTracker(
                        options)
            if options.defaultAnchorMode():
                # In training it makes sense to see only the last result, by default.
                options.anchorMode(enums.AnchorMode.Final)
            if not optimizer:
                optimizer = torch.optim.SGD(self._user_model.parameters(),
                                            lr=0.01)
            model = _impl.OptimizerWrapper(model, optimizer)
        else:
            if options.defaultAnchorMode():
                # In inference it makes sense to see all the results, by default.
                options.anchorMode(enums.AnchorMode.All)
            assert options.Training.gradient_accumulation == 1, (
                "Gradient accumulation"
                " should be left to its default value (1) for inference")
            assert not optimizer, "Optimizer should be None for inference"
        self._model = model

        self._host_weights_version = 0
        self._poptorch_version = poptorch_version

        self._executable = None
        self._options = options
        # The args parser needs to be initilialised before the model gets wrapped
        # otherwise we will not be able to retrieve the real arguments list
        self._args_parser = _args_parser.ArgsParser(model)
        # Inputs used to compile the executable
        self._executable_inputs = None
        self._anchor_memory = {}

        # any anchors with unspecified anchor mode should receive the anchor
        # mode used for graph outputs
        for _, anchor in options.anchored_tensors.items():
            if anchor[1]:
                anchor[2] = options.anchor_mode
                if anchor[2] == enums.AnchorMode.EveryN:
                    anchor[3] = options.anchor_return_period

        self._training = training
        if optimizer:
            self._dict_optimizer = _optimizer_attributes.convertOptimizerToDict(
                optimizer, self._attribute_tracker, options)
        else:
            self._dict_optimizer = {}

        self._new_optimizer = optimizer
        self._dict_new_optimizer = self._dict_optimizer
        self._dirty_host_weights = False
        self._trace = None
        self._is_attached = False
        self._profiling = profiling.Channel(
            "poptorch.trainingModel" if self.
            training else "poptorch.inferenceModel")
        self._profiling.instrument(self, "copyWeightsToHost",
                                   "copyWeightsToDevice", "setOptimizer",
                                   "compile", "destroy")
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
                    if parent() and parent()._dirty_host_weights:  # pylint: disable=protected-access
                        logger.debug("Implicit copyWeightsToHost()")
                        parent()._dirty_host_weights = False  # pylint: disable=protected-access
                        parent().copyWeightsToHost()
                        return True
                    return False

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

            # The mere existence of the "__torch_function__" results in a
            # "__getattribute__" call and hence weight copying if required.
            # "check_has_torch_function" and "handle_torch_function_getter"
            # in the Pytorch source code may explain this.
            # Without this, the weights will not be copied in certain
            # situations such as torch.equal(a, b).
            class PoptorchParameter(torch.nn.Parameter):
                def __getattribute__(self, name):
                    if parent():
                        parent()._user_model.copyWeightsToHostIfNeeded()  # pylint: disable=protected-access

                    return object.__getattribute__(self, name)

                def __torch_function__(self, func, types, args=(),
                                       kwargs=None):
                    if kwargs is None:
                        kwargs = {}
                    return super().__torch_function__(func, types, args,
                                                      kwargs)

            for p in self._user_model.parameters():
                p.__class__ = PoptorchParameter

            class PoptorchBuffer(torch.Tensor):
                def __getattribute__(self, name):
                    if parent():
                        parent()._user_model.copyWeightsToHostIfNeeded()  # pylint: disable=protected-access

                    return super().__getattribute__(name)

                def __torch_function__(self, func, types, args=(),
                                       kwargs=None):
                    if kwargs is None:
                        kwargs = {}
                    return super().__torch_function__(func, types, args,
                                                      kwargs)

            for b in self._user_model.buffers():
                if b.__class__ != torch.Tensor:
                    raise RuntimeError("All buffers must be an instance of " +
                                       "torch.Tensor")
                b.__class__ = PoptorchBuffer

            # __getattr__ and __getattribute__ are attributes, not methods,
            # unfortunately we cannot just replace them in the model object: we
            # have to create a wrapper class
            # and change the object's class.
            PoptorchModel.__name__ = "Poptorch%s" % type(
                self._user_model).__name__
            self._user_model.__class__ = PoptorchModel
            # Register custom function to copy / serialize wrappers
            copyreg.pickle(PoptorchModel, _impl.pickleUnwrapObject)
            copyreg.pickle(PoptorchParameter, _impl.pickleUnwrapObject)
            copyreg.pickle(PoptorchBuffer, _impl.pickleUnwrapObject)

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

    def __getattr__(self, attr):
        return getattr(self._user_model, attr)

    @property
    def model(self) -> 'torch.nn.Module':
        """Access the wrapped Torch model."""
        return self._user_model

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

    # Copy weights from the device into the memory of the model given on wrapper creation.
    def copyWeightsToHost(self) -> None:
        """ Updates the parameters used in `model` with the weights stored on device.
        (The weights in ``model.parameters()``)
        """

        if not self.isCompiled():
            raise RuntimeError(NO_EXECUTABLE_ERR)

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
            raise RuntimeError(NO_EXECUTABLE_ERR)

        # Don't trigger a copyToHost by accessing `named_parameters`
        self._dirty_host_weights = False

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
        self._new_optimizer = optimizer
        self._dict_new_optimizer = _optimizer_attributes.convertOptimizerToDict(
            optimizer, self._attribute_tracker, self._options)

    def _compileWithTrace(self, trace_args):
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
        # Note: in single process execution or if the cache is disabled
        # should_compile will always be False.
        with _impl.distributedCacheLock(self._model,
                                        self._options) as should_compile:
            # Only the first process should compile
            if should_compile:
                self._executable = poptorch_core.compileWithTrace(*trace_args)

        # In distributed execution mode:
        # At that point only the first process will have a compiled executable:
        # trigger the compilation process in all the other processes.
        if not self.isCompiled():
            self._executable = poptorch_core.compileWithTrace(*trace_args)

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

    def _compile(self, in_tensors):
        with self._profiling.tracepoint("modelCompilation"):
            in_tensors_trace_view, has_converted_any_half, narrow_tensor_fn = \
                    self._preprocessGraph(
                        in_tensors)

            # Compile the poplar executable based on the batchsize.
            if self._options.Jit.trace_model:
                trace_args = self._trace_model_and_get_compile_args(
                    in_tensors, in_tensors_trace_view, has_converted_any_half,
                    narrow_tensor_fn)

                self._compileWithTrace(trace_args)
            else:
                logger.info('Compiling the model using scripting')
                self._trace = torch.jit.script(self._model)
                graph_inputs = list(self._trace.graph.inputs())
                for graph_input, arg_in in zip(
                        graph_inputs[1:], in_tensors_trace_view.asTuple()):
                    if isinstance(arg_in, torch.Tensor):
                        graph_input.inferTypeFrom(arg_in)

                parameters = {
                    **dict(self._trace.named_parameters()),
                    **dict(self._trace.named_buffers())
                }

                # pylint: disable=protected-access
                self._executable = poptorch_core.compileWithScript(
                    self._trace._c, self._trace.graph, parameters,
                    in_tensors_trace_view.asTuple(), self._options.toDict(),
                    self._training, accessAttributes,
                    list(self._options.anchored_tensors.values()))

            self._is_attached = self.isAttachedToDevice()

            if self._is_attached:
                # Upload the weights to the IPU
                self.copyWeightsToDevice()

    def _preprocessGraph(self, in_tensors):
        with self._profiling.tracepoint("graphPreprocessing"):
            self._executable_inputs = in_tensors.clone()

            # Input will be in form of [BatchSize* BatchPerStep, ...] so we
            # should slice it up so we compile by the batch size alone.
            extra_poplar_batch_dims = self._options.device_iterations * \
                self._options.replication_factor * \
                self._options.Training.gradient_accumulation

            # There are two concepts of batch size. First is the "model" batch size then there is the
            # concept of batching at the popart level. Here we divide by the popart batch size so the
            # trace "sees" the model batch size but when we call execute we pass the full batch and popart
            # will partition it up.
            in_tensors_trace_view = in_tensors.clone()

            def narrowTensor(tensor):
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
                    "documentation.") % (
                        tensor.shape, b_size, self._options.device_iterations,
                        self._options.replication_factor,
                        self._options.Training.gradient_accumulation,
                        extra_poplar_batch_dims)
                return tensor.narrow(0, 0, b_size // extra_poplar_batch_dims)

            in_tensors_trace_view.forEach(narrowTensor)

            def remove_require_grad(tensor):
                if not isinstance(tensor, torch.Tensor):
                    return tensor

                if tensor.requires_grad:
                    tensor = tensor.detach()
                    logger.warning("Input tensor has requires_grad=True set."
                                   "This tensor will be detached.")

                return tensor

            in_tensors_trace_view.forEach(remove_require_grad)

            # Normal bools don't get captured in python.
            has_converted_any_half = [False]

            def possiblyConvertFromHalf(tensor):
                if isinstance(tensor,
                              torch.Tensor) and tensor.dtype == torch.half:
                    has_converted_any_half[0] = True
                    return tensor.float()
                return tensor

            in_tensors_trace_view.forEach(possiblyConvertFromHalf)

            poptorch_core.processPrecisionOptions(self._options.Precision)

        return in_tensors_trace_view, has_converted_any_half, narrowTensor

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

    def loadExecutable(self, filename: str) -> None:
        """Load an executable previously generated using
        :py:meth:`~poptorch.PoplarExecutor.compileAndExport`
        """
        with self._profiling.tracepoint("loadExecutable"):
            data, exe_offset = _poptorch_data.parse(filename,
                                                    self._poptorch_version)
            in_tensors_trace_view, has_converted_any_half, narrow_tensor_fn = \
                    self._preprocessGraph(
                        data.executable_inputs)
            trace_args = self._trace_model_and_get_compile_args(
                data.executable_inputs, in_tensors_trace_view,
                has_converted_any_half, narrow_tensor_fn)
            self._executable = poptorch_core.processTraceAndImportExecutable(
                *trace_args, filename, exe_offset)
            self._is_attached = self.isAttachedToDevice()

            if self._is_attached:
                # Upload the weights to the IPU
                self.copyWeightsToDevice()

    def compileAndExport(self,
                         filename: str,
                         *args: List['torch.Tensor'],
                         export_model: bool = True,
                         **kwargs: Dict[str, 'torch.Tensor']):
        """Precompile an executable and save it to file.

        ``args`` and ``kwargs`` are the same arguments as the wrapped PyTorch
        ``model.__call__``

        :param str filename: Where to save the compiled executable.
        :param bool export_model: If `True` the Torch model will be saved in
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
        in_tensors = self._args_parser(args, kwargs, False)
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
            logger.warning(
                "compileAndExport: %s is a directory, saving model to %s",
                dirname, filename)

        if export_model:
            data = _poptorch_data.PoptorchData(self._poptorch_version,
                                               in_tensors, self._options,
                                               self._training, self.model,
                                               self._new_optimizer)
        else:
            data = _poptorch_data.PoptorchData(self._poptorch_version,
                                               in_tensors)
        with open(filename, "wb") as f:
            pickle.dump(data, f)
            f.close()
        assert self._options.Jit.trace_model, (
            "compileAndExport not supported for"
            " torch script")
        with self._profiling.tracepoint("compileAndExport"):
            in_tensors_trace_view, has_converted_any_half, narrow_tensor_fn = \
                    self._preprocessGraph(
                        in_tensors)
            trace_args = self._trace_model_and_get_compile_args(
                in_tensors, in_tensors_trace_view, has_converted_any_half,
                narrow_tensor_fn)
            poptorch_core.compileWithTraceAndExport(*trace_args, filename)

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
        # If this is an inference model: check if the same model is not being
        # trained on a different IPU.
        # If it is: make sure the weights are updated.
        if not self._training:
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
        # Execute the poplar executable with the full size (batch * device interations)
        with self._profiling.tracepoint("modelExecution"):
            if self._dict_new_optimizer and \
                    self._dict_new_optimizer != self._dict_optimizer:
                self._dict_optimizer = self._dict_new_optimizer
                output = poptorch_core.execute(self._executable,
                                               in_tensors.asTuple(),
                                               self._dict_optimizer)
            else:
                output = poptorch_core.execute(self._executable,
                                               in_tensors.asTuple(), {})

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

        if self._training:
            self._dirty_host_weights = True

        if len(output) == 0:
            return None
        if len(output) > 1:
            return output
        return output[0]

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

        :param str from_event: key of starting performance counter
        :param from_reduce: reduction function for starting counters
        :param str to_event: key of ending performance counter
        :param to_reduce: redunction function for ending counters
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
        # due to other options such as gradient accumulation and anchor modes.
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
            raise RuntimeError(
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

    def _trace_model_and_get_compile_args(self, in_tensors,
                                          in_tensors_trace_view,
                                          has_converted_any_half,
                                          narrow_tensor_fn):
        logger.info('Compiling the model using tracing')
        # CPU tracing doens't work for half types. We need to convert all half
        # layers to float, run tracing and revert the types to their original.
        half_layers = set()
        all_layers = list(self._model.named_modules())

        # iterate in reverse to process inner layers first
        for (name, layer) in reversed(all_layers):
            any_is_half = False
            for param in layer.parameters():
                if param.dtype == torch.half:
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

        # Override half so user can use it in their models.
        def NewHalf(tensor):
            return _impl.internal_cast(tensor, torch.half)

        # Store the old half so it can be restored.
        old_half = torch.Tensor.half
        torch.Tensor.half = NewHalf

        # Trace only a copy to avoid updating original weights during compilation.
        temp_model = copy.deepcopy(self._model.state_dict())

        added_dummy_output = False

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

        # Restore half to its old meaning.
        torch.Tensor.half = old_half

        # Revert the traced copy to the inital weights.
        self._trace.load_state_dict(temp_model)
        self._model.load_state_dict(temp_model)

        # Restore to non-IPU codepath.
        _impl.setIpuContext(False)

        self._options._execution_strategy.onEndTracing()
        self._RestoreInputs(in_tensors_backup, in_tensors_trace_view_tuple)

        # Some of the trace layers of tuple float should be of type half.
        # The following works because the iterator is hierarchic,
        # yielding containers before contents.
        for name, layer in self._trace.named_modules():
            if name in half_layers:
                layer.half()

        # Convert back the original model as well.
        for name, layer in self._model.named_modules():
            if name in half_layers:
                layer.half()

        # We need to track the parameters from the traced model as this is what the C++ graph sees.
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
            in_tensors_as_half.forEach(narrow_tensor_fn)

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
            raise RuntimeError(NO_EXECUTABLE_ERR)

        if self._training:
            self.copyWeightsToHostIfNeeded()

        if not self._is_attached:
            raise RuntimeError("Device is not attached")

        poptorch_core.detachFromDevice(self._executable)
        self._is_attached = False

    def attachToDevice(self) -> None:
        """Attach to target device. Before calling this function, the device
        must be detached and the model compiled."""
        if not self.isCompiled():
            raise RuntimeError(NO_EXECUTABLE_ERR)
        assert self._options.connection_type != enums.ConnectionType.Never, (
            "Trying to attach to an offline device"
            " (ConnectionType.Never)")

        if self._is_attached:
            raise RuntimeError("Device is already attached")

        poptorch_core.attachToDevice(self._executable)
        poptorch_core.loadEngineAndConnectStreams(self._executable)
        self._is_attached = True
        # Upload the weights to the IPU
        self.copyWeightsToDevice()
        # PopART save / restore the optimizer state with the weight,
        # but parameters  need to be re-uploaded
        self._dict_optimizer = {}

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
        raise RuntimeError("Unsupported input type or condition.")
