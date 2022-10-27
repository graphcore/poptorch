# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import collections
import functools
import warnings
from typing import Callable, List, Optional

import torch
from . import enums, poptorch_core, _impl, accessAttributes, CompilerOptions
from ._args_parser import ArgsParser
from ._utils import flattenTensorStructure, reconstructTensorStructure, isOnIpu
from .options import Options
from .optim import Optimizer


class IPUScope:
    def __init__(self,
                 inputs: List['torch.Tensor'],
                 model: Optional['torch.nn.Module'] = None,
                 options: Optional['poptorch.Options'] = None,
                 training: bool = False,
                 dict_optimizer: Optional[dict] = None,
                 optimizer: Optimizer = None,
                 compile_using=enums.Compiler.PopART,
                 skip_compilation=False,
                 per_replica_params: Optional[dict] = None):

        if not isinstance(inputs, (list, tuple)):
            raise ValueError("You can only pass a list or tuple as the "
                             "inputs argument to IPUScope.")

        # Check that the inputs are a valid type
        for tensor in inputs:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("You can only pass torch.Tensors as inputs "
                                 "to IPUScope.")

            if tensor.is_sparse:
                raise ValueError("You cannot pass sparse tensors as inputs "
                                 "to IPUScope.")

        if model and not isinstance(model, torch.nn.Module):
            raise ValueError("model must inherit from torch.nn.Module")

        self._executable = None
        self._options = options or Options()
        if self._options.defaultOutputMode():
            self._options = self._options.outputMode(enums.OutputMode.All)

        self._training = training
        self._optimizer = optimizer
        self._dict_optimizer = {} if dict_optimizer is None else dict_optimizer

        self._compile_using = compile_using

        self._model = model
        self._cpu_params = {}
        self._cpu_buffers = {}

        # A map of parameters and buffers (tensors) on the CPU which share
        # the same python id, to the earliest tensor.
        self._cpu_aliases = {}

        self._outputs_structure = None
        self._upload_weights = True
        self._skip_compilation = skip_compilation
        self._per_replica_params = per_replica_params

        mlir_compiler_options = CompilerOptions()
        mlir_compiler_options.source_location_excludes = self._options._source_location_excludes  # pylint: disable=line-too-long

        # Create the graph. Future captured calls will be written into this
        # graph behind the scenes.
        poptorch_core.createGraph(
            poptorch_core.TracingMode(self._compile_using), inputs,
            mlir_compiler_options)

        self._old_addresses = {}

    def register_optimizer_groups(self):
        # The optimizer was created using the CPU model, therefore it points
        # at CPU tensors.  We need to remap those to IPU tensors.
        # IPUContext moved 'model' to the IPU, therefore we need to join the
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
        if self._model and self._optimizer:
            cpu_tensors = {
                **self._cpu_buffers,
                **self._cpu_params,
            }
            ipu_tensors = _impl.getBufferAndParameterTensors(self._model)
            cpu_to_ipu = {
                cpu_tensors[n].data_ptr(): ipu
                for n, ipu in ipu_tensors.items()
            }
            for index, group in enumerate(self._optimizer.param_groups):
                torch.ops.poptorch.optimizer_group(
                    index,
                    [cpu_to_ipu[cpu.data_ptr()] for cpu in group["params"]])

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

    # Start capturing calls.
    def __enter__(self):
        # Move the model parameters to the ipu and take a copy to load the
        # originals back once this has finished
        if self._model:
            self._cpu_params = dict(self._model.named_parameters())
            self._cpu_buffers = dict(self._model.named_buffers())
            cpu_state = self._model.state_dict(keep_vars=True)

            # We need to remove the PoptorchBuffer and PoptorchParam annotations
            # before compiling the model. In addition, we must unwrap the whole
            # model to prevent IPU to CPU copies when accessing the state_dict.
            _impl.unwrapModelIfNecessary(self._model)

            if self._per_replica_params is not None:
                for name, param in self._cpu_params.items():
                    if name in self._per_replica_params:
                        if param.shape == torch.Size([]):
                            raise _impl.createPoptorchError(
                                "Scalars cannot be passed as per-replica "
                                "weight tensor values")
                        param_tensor = param.narrow(0, 0, 1).squeeze(dim=0)
                        setattr(*self._get_module_and_name(name),
                                torch.nn.Parameter(param_tensor))
            # TODO(T61576) We currently use a state machine to determine if
            # tensors are inputs or parameters.
            # We need to find a better solution.
            d = torch.device("xla:0")
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
            aliases = [v for v in tensors.values() if len(v) > 1]
            for a in aliases:
                # NB original matches that in model.named_x() as both this as
                # model.state_dict() loop he same  OrderedDicts in same order
                # and the named versions return only the first instances
                original = a[0]

                for other in a[1:]:
                    setattr(*self._get_module_and_name(other), state[original])
                    self._cpu_aliases[other] = original

            # Map named unique parameters and buffers on the IPU.
            params = dict(self._model.named_parameters())

            poptorch_core.mapParamsToNames(tuple(params.keys()),
                                           tuple(params.values()))

            buffers = dict(self._model.named_buffers())

            poptorch_core.mapParamsToNames(tuple(buffers.keys()),
                                           tuple(buffers.values()))

            self._old_addresses = _impl.getBufferAndParameterAddresses(
                self._model)

            if self._per_replica_params is not None:
                for name, param in self._cpu_params.items():
                    if name in self._per_replica_params:
                        poptorch_core.setPerReplica(
                            name, param, *self._per_replica_params[name])

        poptorch_core.startDispatch()
        _impl.setDispatchTracing(True)
        _impl.setIpuContext(True)
        self._options._execution_strategy.onStartTracing()  # pylint: disable=protected-access

        self.register_optimizer_groups()

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

            def _set_param(k, v):
                setattr(*self._get_module_and_name(k), self._cpu_params[v])

            for k in self._cpu_params:
                self._cpu_params[k].__class__ = torch.nn.Parameter
                _set_param(k, k)

            # Restore aliased parameters/buffers which will not be represented
            # in self._cpu_params or self._cpu_buffers
            for k, v in self._cpu_aliases.items():
                _set_param(k, v)

            for k in self._cpu_buffers:
                setattr(*self._get_module_and_name(k), self._cpu_buffers[k])

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
            self._executable = poptorch_core.compileMLIR()
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
            output = self._executable.execute(args)
            self._executable.copyWeightsToHostIfNeeded()

        if self._outputs_structure is not None:
            output = reconstructTensorStructure(self._outputs_structure,
                                                output)
        return output

    def outputs(self, tensors):
        self._outputs_structure = tensors
        output = flattenTensorStructure(tensors)

        for x in output:
            if not isOnIpu(x):
                warnings.warn("Output expected to be on the IPU but is on %s" %
                              x.device.type)

        output = [
            out.int() if out.dtype == torch.long and isOnIpu(out) else out
            for out in output
        ]
        output = [
            out.float() if out.dtype == torch.double and isOnIpu(out) else out
            for out in output
        ]
        poptorch_core.startOutputsMove()
        output = [out.cpu() for out in output]
        poptorch_core.endOutputsMove()

        poptorch_core.finalizeGraph()


# TODO(T45467): Modify this wrapper so that sentinel dispatch occurs on each
#               call after compilation so that changes to wrapped values
#               can be picked up
class _IPUContext:
    def __init__(self, func, compiler, options, training, optimizer,
                 dict_optimizer, model, per_replica_params):
        functools.update_wrapper(self, func)
        self.func = func
        self.ipu = None
        self.compiler = compiler
        self.options = options or Options()
        self.training = training
        self.optimizer = optimizer
        self.dict_optimizer = dict_optimizer
        self.model = model
        self.per_replica_params = per_replica_params

    def compile(self, *args, **kwargs):
        return self._compileOrLoadExecutable(args, kwargs)

    def loadExecutable(self, filename, *args, **kwargs):
        return self._compileOrLoadExecutable(args, kwargs, filename)

    @_impl.destroyDispatcherOnExit
    def _compileOrLoadExecutable(self, args, kwargs, filename=None):
        module_namescope = None
        # TODO(T66133) Add support for name scopes in MLIR backend.
        # pylint: disable=protected-access
        if self.model is not None and \
                self.compiler == enums.Compiler.PopART and \
                self.options._module_namescope_enabled:
            module_namescope = _impl.NameScopeHook(self.model)

        tensor_args = flattenTensorStructure((args, kwargs))
        with IPUScope(tensor_args,
                      model=self.model,
                      options=self.options,
                      training=self.training,
                      optimizer=self.optimizer,
                      dict_optimizer=self.dict_optimizer,
                      compile_using=self.compiler,
                      skip_compilation=filename is not None,
                      per_replica_params=self.per_replica_params) as ipu:

            for idx, t in enumerate(tensor_args):
                if t.requires_grad:
                    raise _impl.createPoptorchError(
                        "An input tensor to an IPU model can not have "
                        f"requires_grad set to True, however input {idx} "
                        f"does: {t}\nYou can set requires_grad=True from "
                        "within the model as an alternative, and return "
                        "gradients as outputs to your model, if required.")

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

        if module_namescope:
            module_namescope.remove()

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
               optimizer: Optimizer = None,
               dict_optimizer: Optional[dict] = None,
               model: Optional['torch.nn.Module'] = None,
               per_replica_params: Optional[dict] = None):
    if dict_optimizer is None:
        dict_optimizer = {}

    # If our decorator is passed any explicit keyword arguments, "func"
    # will be None so we need to return a new decorator with the keyword
    # arguments hardcoded in
    if func is None:

        def wrapper(f):
            return _IPUContext(f, compiler, options, training, optimizer,
                               dict_optimizer, model, per_replica_params)

        return wrapper
    # Otherwise the decorator has no extra args: just pass the
    # default arguments
    return _IPUContext(func, compiler, options, training, optimizer,
                       dict_optimizer, model, per_replica_params)


class _IPUSession:
    """
    Internal use only.

    Context manager for creating a session in which to build an IPU graph.
    """

    def __init__(self, compiler_options: CompilerOptions):
        self._compiler_options = compiler_options

        # Create the graph. Future captured calls will be written into this
        # graph behind the scenes.
        poptorch_core.createGraph(
            poptorch_core.TracingMode(enums.Compiler.MLIR), [],
            self._compiler_options)

    def run(self, func: Callable, args: ArgsParser.Args):
        poptorch_core.startDispatch()
        _impl.setDispatchTracing(True)
        _impl.setIpuContext(True)
        poptorch_core.promoteArgsAsInputs(args.asPackedFlatTuple())

        excepted = False
        out = None
        try:
            out = func(*args.args, **args.kwargs)
        except Exception as exc:
            excepted = True
            raise exc
        finally:
            if not excepted and out is not None:
                flattened = flattenTensorStructure(out)
                poptorch_core.promoteOutputs(flattened)

            poptorch_core.finalizeGraph()
            _impl.setIpuContext(False)
            _impl.setDispatchTracing(False)
            poptorch_core.endDispatch(excepted)

        return out

    def compile(self):
        """
        Compile the graph built in this IPUSession, and return an executable,
        on which you can call `executable.execute()`.
        """
        return poptorch_core.compileMLIR()


def ipu_wrapper(_func: Optional[Callable] = None,
                *,
                compiler_options: CompilerOptions = CompilerOptions()):
    """Function decorator which compiles the IPU graph contained within.

    The previously compiled executable are kept in a cache and compilation is
    avoided if the shapes do not change from a previous run.

    Usage:
    ```
    @ipu_wrapper
    def my_ipu_func(a, b):
        return a @ b

    a = ...
    b = ...
    res = my_ipu_func(a.to("ipu"), b.to("ipu"))

    print("Result of a @ b:", res.to("cpu"))
    ```

    Arguments:
    options -- poptorch.CompilerOptions structure to configure runtime
               parameters.
    """

    def decorator(func: Callable):
        cache = None
        args_parser: ArgsParser = ArgsParser(func, tracing=False)
        compiled_args: Optional[ArgsParser.Args] = None
        output_structure = None

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal cache, args_parser, compiled_args, output_structure, \
                compiler_options

            current_args = args_parser(args, kwargs)
            flat_args = current_args.asPackedFlatTuple()
            if len(flat_args) == 0:
                raise _impl.createPoptorchError(
                    "No tensor inputs passed to `ipu_wrapper`-wrapped "
                    "function. At least one input to an this function must be "
                    "a `torch.Tensor`.")

            if compiled_args:
                try:
                    compiled_args.validateInputs(current_args)
                    out = cache.execute(flat_args)
                    return reconstructTensorStructure(output_structure, out)
                except poptorch_core.Error:
                    pass

            sess = _IPUSession(compiler_options)
            output_structure = sess.run(func, current_args)

            executable = sess.compile()
            out = executable.execute(flat_args)

            # Save this executable as compiled with these args.
            cache = executable
            compiled_args = current_args

            return reconstructTensorStructure(output_structure, out)

        return wrapper

    if _func is None:
        return decorator

    return decorator(_func)
