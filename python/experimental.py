# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import functools
import types
from typing import Dict, List, Optional

import torch
from . import enums, poptorch_core, _impl, accessAttributes
from ._utils import unrollTensorList
from .options import Options


class IPUScope:
    def __init__(
            self,
            inputs: List['torch.Tensor'],
            parameters_and_buffers: Optional[Dict[str, 'torch.Tensor']] = None,
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

        self._executable = None
        self._options = options or Options()
        if self._options.defaultOutputMode():
            self._options = self._options.outputMode(enums.OutputMode.All)

        self._compile_using = compile_using

        if isinstance(parameters_and_buffers, types.GeneratorType):
            parameters_and_buffers = {
                **dict(parameters_and_buffers),
            }

        if parameters_and_buffers is None:
            self._params_and_buffers = {}
        else:
            self._params_and_buffers = parameters_and_buffers

        param_list = list(self._params_and_buffers.values())

        self._outputs = []
        self._inputs = []
        self._upload_weights = True

        with torch.no_grad():
            self._inputs = [t.clone() for t in inputs]

        # Create the graph. Futured captured calls will be written into this graph behind the scenes.
        poptorch_core.createGraph(
            poptorch_core.TracingMode(self._compile_using), inputs, param_list)

    # Start capturing calls.
    def __enter__(self):
        poptorch_core.startDispatch()
        _impl.setIpuContext(True)
        self._options._execution_strategy.onStartTracing()  # pylint: disable=protected-access
        return self

    # Exit the scope. Compile graph and stop capturing call
    def __exit__(self, exc_type, value, traceback):
        self._options._execution_strategy.onEndTracing()  # pylint: disable=protected-access
        _impl.setIpuContext(False)
        # Turn off the dispatcher.
        poptorch_core.endDispatch()

        # Dispatch stopped because of an exception: don't try to compile
        # the graph.
        if exc_type is not None:
            return

        # Compile for IPU.
        if self._compile_using == enums.Compiler.PopART:
            # Compile the captured graph using PopART.
            self._executable = poptorch_core.compileWithManualTracing(
                self._inputs, list(self._params_and_buffers.values()),
                list(self._params_and_buffers.keys()), self._options.toDict(),
                accessAttributes)
        else:
            # Compile the captured graph using MLIR.
            self._executable = poptorch_core.compileWithMlir()

    def __call__(self, *args):
        if self._upload_weights:
            if self._compile_using == enums.Compiler.PopART:
                poptorch_core.copyWeightsToDevice_impl(
                    self._executable, tuple(self._params_and_buffers.keys()),
                    tuple(self._params_and_buffers.values()))
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

        if len(output) == 1:
            return output[0]

        return output

    def outputs(self, tensors):
        # We don't want to catch anything in here.
        poptorch_core.endDispatch()

        with torch.no_grad():
            for tensor in tensors:
                if tensor.dtype == torch.torch.long:
                    self._outputs.append(tensor.int())
                else:
                    self._outputs.append(tensor.clone())

        poptorch_core.markOutputs(tensors, self._outputs)

        # Turn dispatch back on.
        poptorch_core.startDispatch()


# TODO(T45467): Modify this wrapper so that sentinel dispatch occurs on each
#               call after compilation so that changes to wrapped values
#               can be picked up
class _IPUContext:
    def __init__(self, func, params, compiler, options):
        functools.update_wrapper(self, func)
        self.func = func
        self.ipu = None
        self.compiler = compiler
        self.params = params
        self.options = options

    def __call__(self, *args, **kwargs):
        # Collect all tensor arguments and pass them to IPUScope
        tensor_args = unrollTensorList((*args, *kwargs.values()))
        if self.ipu is None:
            with IPUScope(tensor_args,
                          parameters_and_buffers=self.params,
                          options=self.options,
                          compile_using=self.compiler) as ipu:
                result = self.func(*args, **kwargs)
                tensor_outputs = unrollTensorList(result)
                if len(tensor_outputs) == 0:
                    raise TypeError("A function marked with IPUContext must "
                                    "return one or more tensors.")
                ipu.outputs(tensor_outputs)
            self.ipu = ipu
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
# You can also pass in parameters and poptorch options:
#
# 1.     ipu_func = IPUContext(func, parameters_and_buffers=..., options=...)(inputs)
#
# 2.     @IPUContext(parameters_and_buffers=..., options=...)
#        def ipu_func(inputs):
#            ...
#            return outputs
#
# Note that the function being wrapped MUST take the graph inputs as inputs to the
# function, and MUST return the outputs of the graph. Any non-tensor inputs will
# be ignored by the graph. The function outputs must consist of tensors only
def IPUContext(
        func=None,
        *,
        parameters_and_buffers: Optional[Dict[str, 'torch.Tensor']] = None,
        compiler=enums.Compiler.MLIR,
        options: Optional['poptorch.Options'] = None):
    # If our decorator is passed any explicit keyword arguments, "func"
    # will be None so we need to return a new decorator with the keyword
    # arguments hardcoded in
    if func is None:

        def wrapper(f):
            return _IPUContext(f, parameters_and_buffers, compiler, options)

        return wrapper
    # Otherwise the decorator has no extra args: just pass the
    # default arguments
    return _IPUContext(func, parameters_and_buffers, compiler, options)
