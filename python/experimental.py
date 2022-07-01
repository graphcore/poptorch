# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import functools
import warnings
from typing import List, Optional

import torch
from . import enums, poptorch_core, _impl, accessAttributes
from ._utils import flattenTensorStructure, reconstructTensorStructure, isOnIpu
from .options import Options


class IPUScope:
    def __init__(self,
                 inputs: List['torch.Tensor'],
                 model: Optional['torch.nn.Module'] = None,
                 options: Optional['poptorch.Options'] = None,
                 training: bool = False,
                 dict_optimizer: Optional[dict] = None,
                 compile_using=enums.Compiler.PopART,
                 skip_compilation=False):

        if not isinstance(inputs, (list, tuple)):
            raise ValueError("You can only pass a list or tuple as the " +
                             "inputs argument to IPUScope.")

        # Check that the inputs are a valid type
        for tensor in inputs:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("You can only pass torch.Tensors as inputs " +
                                 "to IPUScope.")

            if tensor.is_sparse:
                raise ValueError("You cannot pass sparse tensors as inputs " +
                                 "to IPUScope.")

        if model and not isinstance(model, torch.nn.Module):
            raise ValueError("model must inherit from torch.nn.Module")

        self._executable = None
        self._options = options or Options()
        if self._options.defaultOutputMode():
            self._options = self._options.outputMode(enums.OutputMode.All)

        self._training = training
        self._dict_optimizer = {} if dict_optimizer is None else dict_optimizer

        self._compile_using = compile_using

        self._model = model
        self._cpu_params = {}
        self._cpu_buffers = {}

        self._outputs = []
        self._outputs_structure = None
        self._upload_weights = True
        self._skip_compilation = skip_compilation

        # Create the graph. Futured captured calls will be written into this
        # graph behind the scenes.
        poptorch_core.createGraph(
            poptorch_core.TracingMode(self._compile_using), inputs,
            self._options._source_location_excludes)

        self._old_addresses = {}

    # Start capturing calls.
    def __enter__(self):
        # Move the model parameters to the ipu and take a copy to load the
        # originals back once this has finished
        if self._model:
            self._cpu_params = dict(self._model.named_parameters())
            self._cpu_buffers = dict(self._model.named_buffers())

            # We need to remove the PoptorchBuffer and PoptorchParam annotations
            # before compiling the model
            _impl.unwrapModelIfNecessary(self._model)

            # TODO(T61576) We currently use a state machine to determine if
            # tensors are inputs or parameters.
            # We need to find a better solution.
            d = torch.device("xla:0")
            poptorch_core.startParametersMove()
            self._model.to(d)
            poptorch_core.endParametersMove()

            params = dict(self._model.named_parameters())

            poptorch_core.mapParamsToNames(tuple(params.keys()),
                                           tuple(params.values()))

            buffers = dict(self._model.named_buffers())

            poptorch_core.mapParamsToNames(tuple(buffers.keys()),
                                           tuple(buffers.values()))

            self._old_addresses = _impl.getBufferAndParameterAddresses(
                self._model)

        poptorch_core.startDispatch()
        _impl.setDispatchTracing(True)
        _impl.setIpuContext(True)
        self._options._execution_strategy.onStartTracing()  # pylint: disable=protected-access
        return self

    # Exit the scope. Compile graph and stop capturing call
    def __exit__(self, exc_type, value, traceback):
        self._options._execution_strategy.onEndTracing()  # pylint: disable=protected-access
        _impl.setIpuContext(False)
        _impl.setDispatchTracing(False)
        # Turn off the dispatcher.
        poptorch_core.endDispatch(exc_type is not None)

        # Reload the cpu model state
        if self._model:
            # Get the buffer and parameter addresses after the model has ran
            # but before resetting the model back to the cpu
            new_addresses = _impl.getBufferAndParameterAddresses(self._model)

            def get_model_and_name(n):
                m = self._model
                name = n
                sn = n.rpartition(".")
                if sn[1] == ".":
                    m = m.get_submodule(sn[0])
                    name = sn[2]
                return m, name

            for k in self._cpu_params:
                self._cpu_params[k].__class__ = torch.nn.Parameter
                setattr(*get_model_and_name(k), self._cpu_params[k])
            for k in self._cpu_buffers:
                setattr(*get_model_and_name(k), self._cpu_buffers[k])

            # Re-install the Poptorch annotations for buffers and parameters
            _impl.rewrapModelIfNecessary(self._model)

            # Check that the buffer and parameter addresses haven't been changed
            # in the model
            # Note: this is done after resetting the model back to the cpu so
            # that errors thrown by this don't stop the model being in a valid
            # state
            _impl.errorOnBufferOrParameterAddressChanges(
                self._old_addresses, new_addresses)

        # Dispatch stopped because of an exception: don't try to compile
        # the graph.
        if exc_type is not None:
            return False

        if self._skip_compilation:
            return True

        # Compile for IPU.
        if self._compile_using == enums.Compiler.PopART:
            # Compile the captured graph using PopART.
            self._executable = poptorch_core.compileWithManualTracing(
                self._options.toDict(), accessAttributes, self._training,
                self._dict_optimizer,
                list(self._options.anchored_tensors.values()))
        else:
            # Compile the captured graph using MLIR.
            self._executable = poptorch_core.compileWithMLIR()
        return True

    def loadExecutable(self, filename):
        if self._compile_using == enums.Compiler.PopART:
            # Compile the captured graph using PopART.
            self._executable = poptorch_core.processDispatchAndImportExecutable(
                self._options.toDict(), accessAttributes, self._training,
                self._dict_optimizer,
                list(self._options.anchored_tensors.values()), filename)
        else:
            raise _impl.createPoptorchError("Not supported: can't deserialize "
                                            "MLIR executables")

    def __call__(self, *args):
        if self._upload_weights:
            if self._compile_using == enums.Compiler.PopART:
                state = {**self._cpu_params, **self._cpu_buffers}
                poptorch_core.copyWeightsToDevice_impl(self._executable,
                                                       tuple(state.keys()),
                                                       tuple(state.values()))
            else:
                self._executable.weightsToDevice()
            self._upload_weights = False

        # Otherwise run popart.
        if self._compile_using == enums.Compiler.PopART:
            # Run via PopART.
            output = poptorch_core.execute(self._executable, args)
        elif self._compile_using == enums.Compiler.MLIR:
            # Run via the MLIR compiled binary.
            self._executable.execute(args)
            output = self._outputs

        if self._outputs_structure is not None:
            output = reconstructTensorStructure(self._outputs_structure,
                                                output)
        return output

    def outputs(self, tensors):
        self._outputs_structure = tensors
        self._outputs = list(flattenTensorStructure(tensors))

        for x in self._outputs:
            if not isOnIpu(x):
                warnings.warn("Output expected to be on the IPU but is on %s" %
                              x.device.type)

        self._outputs = [
            out.int() if out.dtype == torch.long and isOnIpu(out) else out
            for out in self._outputs
        ]
        self._outputs = [
            out.float() if out.dtype == torch.double and isOnIpu(out) else out
            for out in self._outputs
        ]
        self._outputs = [out.cpu() for out in self._outputs]

        poptorch_core.finalizeGraph()


# TODO(T45467): Modify this wrapper so that sentinel dispatch occurs on each
#               call after compilation so that changes to wrapped values
#               can be picked up
class _IPUContext:
    def __init__(self, func, compiler, options, training, dict_optimizer,
                 model):
        functools.update_wrapper(self, func)
        self.func = func
        self.ipu = None
        self.compiler = compiler
        self.options = options
        self.training = training
        self.dict_optimizer = dict_optimizer
        self.model = model

    def compile(self, *args, **kwargs):
        return self._compileOrLoadExecutable(args, kwargs)

    def loadExecutable(self, filename, *args, **kwargs):
        return self._compileOrLoadExecutable(args, kwargs, filename)

    @_impl.destroyDispatcherOnExit
    def _compileOrLoadExecutable(self, args, kwargs, filename=None):
        tensor_args = flattenTensorStructure((args, kwargs))
        with IPUScope(tensor_args,
                      model=self.model,
                      options=self.options,
                      training=self.training,
                      dict_optimizer=self.dict_optimizer,
                      compile_using=self.compiler,
                      skip_compilation=filename is not None) as ipu:
            d = torch.device("xla:0")
            # Move all the inputs to the IPU
            tensor_args = [t.to(d) for t in tensor_args]
            # Re-inject moved tensors in args and kwargs:
            args, kwargs = reconstructTensorStructure((args, kwargs),
                                                      tensor_args)

            result = self.func(*args, **kwargs)
            if result is not None:
                ipu.outputs(result)

        if filename is not None:
            ipu.loadExecutable(filename)

        self.ipu = ipu
        return tensor_args

    def __call__(self, *args, **kwargs):
        # Collect all tensor arguments and pass them to IPUScope
        if self.ipu is None:
            tensor_args = self.compile(*args, **kwargs)
        else:
            tensor_args = flattenTensorStructure((args, kwargs))
        return self.ipu(*tensor_args)


# You can use IPUContext in two different ways:
#
# 1. As an explicit function call:
#        ipu_func = IPUContext(func)(inputs)
#
# 2. As a decorator:
#        @IPUContext
#        def ipu_func(inputs):
#            ...
#            return outputs
#
# You can also pass in a model and poptorch options:
#
# 1. ipu_func = IPUContext(func, model=..., options=...)(inputs)
#
# 2. @IPUContext(model=..., options=...)
#    def ipu_func(inputs):
#        ...
#        return outputs
#
# Note that the function being wrapped MUST take the graph inputs as inputs to
# the function, and MUST return the outputs of the graph. Any non-tensor inputs
# will be ignored by the graph. The function outputs must consist of tensors
# only
def IPUContext(func=None,
               *,
               compiler=enums.Compiler.MLIR,
               options: Optional['poptorch.Options'] = None,
               training: bool = False,
               dict_optimizer: Optional[dict] = None,
               model: Optional['torch.nn.Module'] = None):
    if dict_optimizer is None:
        dict_optimizer = {}

    # If our decorator is passed any explicit keyword arguments, "func"
    # will be None so we need to return a new decorator with the keyword
    # arguments hardcoded in
    if func is None:

        def wrapper(f):
            return _IPUContext(f, compiler, options, training, dict_optimizer,
                               model)

        return wrapper
    # Otherwise the decorator has no extra args: just pass the
    # default arguments
    return _IPUContext(func, compiler, options, training, dict_optimizer,
                       model)
