# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import copy
import functools
from typing import List, Optional

import torch
from . import enums, poptorch_core, _impl, accessAttributes
from ._utils import flattenTensorStructure, reconstructTensorStructure
from .options import Options


class IPUScope:
    def __init__(self,
                 inputs: List['torch.Tensor'],
                 model: Optional['torch.nn.Module'] = None,
                 options: Optional['poptorch.Options'] = None,
                 compile_using=enums.Compiler.PopART):

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

        self._compile_using = compile_using

        self._model = model
        if self._compile_using == enums.Compiler.PopART:
            # PopART requires some CPU pointers to copy to/from the IPU so
            # keep a copy of the CPU buffers.
            self._cpu_model = copy.deepcopy(model)

        self._outputs = []
        self._outputs_structure = None
        self._upload_weights = True

        # Create the graph. Futured captured calls will be written into this
        # graph behind the scenes.
        poptorch_core.createGraph(
            poptorch_core.TracingMode(self._compile_using), inputs,
            self._options._source_location_excludes)

        if self._model:
            # TODO(T61576) We currently use a state machine to determine if
            # tensors are inputs or parameters.
            # We need to find a better solution.
            d = torch.device("xla:0")
            poptorch_core.startParametersMove()
            self._model.apply(lambda l: l.to(d))
            poptorch_core.endParametersMove()

            state = self._model.state_dict()
            poptorch_core.mapParamsToNames(tuple(state.keys()),
                                           tuple(state.values()))

    # Start capturing calls.
    def __enter__(self):
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

        # Dispatch stopped because of an exception: don't try to compile
        # the graph.
        if exc_type is not None:
            return False

        # Compile for IPU.
        if self._compile_using == enums.Compiler.PopART:
            # Compile the captured graph using PopART.
            self._executable = poptorch_core.compileWithManualTracing(
                self._options.toDict(), accessAttributes)
        else:
            # Compile the captured graph using MLIR.
            self._executable = poptorch_core.compileWithMlir()
        return True

    def __call__(self, *args):
        if self._upload_weights:
            if self._compile_using == enums.Compiler.PopART:
                if self._cpu_model is not None:
                    state = self._cpu_model.state_dict()
                else:
                    state = {}
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
        def flatten(x):
            if isinstance(x, (tuple, list)):
                for e in x:
                    yield from flatten(e)
            else:
                if x.device.type != "xla":
                    raise ValueError(
                        "Output expected to be on the IPU but is on %s" %
                        x.device.type)

                yield x

        self._outputs_structure = tensors
        self._outputs = list(flatten(tensors))
        self._outputs = [
            out.int() if out.dtype == torch.long else out
            for out in self._outputs
        ]
        self._outputs = [
            out.float() if out.dtype == torch.double else out
            for out in self._outputs
        ]
        self._outputs = [out.cpu() for out in self._outputs]

        poptorch_core.finalizeGraph()


# TODO(T45467): Modify this wrapper so that sentinel dispatch occurs on each
#               call after compilation so that changes to wrapped values
#               can be picked up
class _IPUContext:
    def __init__(self, func, compiler, options, model):
        functools.update_wrapper(self, func)
        self.func = func
        self.ipu = None
        self.compiler = compiler
        self.options = options
        self.model = model

    def compile(self, *args, **kwargs):
        tensor_args = flattenTensorStructure((args, kwargs))
        with IPUScope(tensor_args,
                      model=self.model,
                      options=self.options,
                      compile_using=self.compiler) as ipu:
            d = torch.device("xla:0")
            # Move all the inputs to the IPU
            tensor_args = [t.to(d) for t in tensor_args]
            # Re-inject moved tensors in args and kwargs:
            args, kwargs = reconstructTensorStructure((args, kwargs),
                                                      tensor_args)

            result = self.func(*args, **kwargs)
            if result is not None:
                ipu.outputs(result)
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
               model: Optional['torch.nn.Module'] = None):
    # If our decorator is passed any explicit keyword arguments, "func"
    # will be None so we need to return a new decorator with the keyword
    # arguments hardcoded in
    if func is None:

        def wrapper(f):
            return _IPUContext(f, compiler, options, model)

        return wrapper
    # Otherwise the decorator has no extra args: just pass the
    # default arguments
    return _IPUContext(func, compiler, options, model)
