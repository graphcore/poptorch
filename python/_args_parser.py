# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import copy
import inspect
from typing import Any, Dict
import torch

# Do not import any poptorch.* here: it will break the poptorch module
from . import _impl
from ._logging import logger
from . import _utils


class ArgsParser:
    class Args:
        def __init__(self):
            self._args = []
            self._arg_names = []
            self._kwargs = {}
            self.first_none = None

        @property
        def args(self):
            return self._args

        @property
        def arg_names(self):
            return self._arg_names

        @property
        def kwargs(self):
            return self._kwargs

        def appendArg(self, arg, name):
            self._args.append(arg)
            self._arg_names.append(name)

        def setNamedArg(self, name, arg):
            self._kwargs[name] = arg

        def popArg(self):
            self._args.pop()
            self._arg_names.pop()

        def clone(self):
            # pylint: disable=protected-access
            clone = ArgsParser.Args()
            clone._args = copy.copy(self._args)
            clone._arg_names = copy.copy(self._arg_names)
            clone._kwargs = copy.copy(self._kwargs)
            clone.first_none = self.first_none
            return clone

        def _forEach(self, data, fn):
            if isinstance(data, (tuple, list)):
                return type(data)(self._forEach(d, fn) for d in data)
            if isinstance(data, dict):
                return {
                    key: self._forEach(value, fn)
                    for key, value in data.items()
                }
            return fn(data)

        def validateInputs(self, inputs):
            end = (
                "\nThis error occurred because the inputs passed at runtime"
                " don't match the inputs used to compile the model.\n"
                "To recompile the model for the new inputs create a new "
                "inferenceModel / trainingModel wrapper or call destroy() on "
                "the curent one and try again.")
            if len(inputs.args) != len(self.args):
                raise _impl.createPoptorchError(
                    "Number of positional arguments mismatch: expected "
                    f"{len(self.args)} arguments but got "
                    f"{len(inputs.args)}.{end}")

            def validate(name, compiled, input, are_named_args=False):
                ctype = type(compiled)
                itype = type(input)
                if ctype != itype:
                    raise _impl.createPoptorchError(
                        f"Type mismatch for {name}: expected "
                        f"{ctype} but got {itype}.{end}")
                if isinstance(compiled, tuple):
                    clen = len(compiled)
                    ilen = len(input)
                    if clen != ilen:
                        raise _impl.createPoptorchError(
                            f"Length mismatch for {name}: "
                            f"expected {clen} elements but got {ilen}.{end}")
                    for i, c in enumerate(compiled):
                        validate(name + f"[{i}]", c, input[i])
                elif isinstance(compiled, dict):
                    expected = set(compiled.keys())
                    provided = set(input.keys())
                    if expected != provided:
                        extra = provided - expected
                        details = []
                        if extra:
                            details.append("Unexpected arguments: " +
                                           ", ".join(sorted(extra)))
                        missing = expected - provided
                        if missing:
                            details.append("Missing arguments: " +
                                           ", ".join(sorted(missing)))
                        raise _impl.createPoptorchError(
                            f"Keys mismatch for {name}: "
                            f"{'. '.join(details)}.{end}")
                    for k, v in compiled.items():
                        if are_named_args:
                            n = k
                        else:
                            n = f"{name}[{k}]"
                        validate(n, v, input[k])

                elif isinstance(compiled, torch.Tensor):
                    if compiled.dtype != input.dtype:
                        raise _impl.createPoptorchError(
                            "Data type "
                            f"mismatch for {name}: expected {compiled.dtype} "
                            f"but got {input.dtype}.{end}")
                    if compiled.shape != input.shape:
                        raise _impl.createPoptorchError(
                            "Shape "
                            f"mismatch for {name}: expected {compiled.shape} "
                            f"but got {input.shape}.{end}")
                else:
                    # If we've got a custom parser then we'll be able to extract
                    # the tensors and validate them as a list.
                    compiled_tensors = _utils.flattenTensorStructure(compiled)
                    if compiled_tensors:
                        input_tensors = _utils.flattenTensorStructure(input)
                        validate(name, tuple(compiled_tensors),
                                 tuple(input_tensors))
                    elif compiled != input:
                        # Other types are compiled in the graph (scalars, etc) and
                        # therefore should be an exact match to the value used to
                        # compile the model.
                        raise _impl.createPoptorchError(
                            f"Value mismatch for {name}: "
                            f"expected {compiled} but got {input}.{end}")

            for i, arg in enumerate(self.args):
                validate(self.arg_names[i], arg, inputs.args[i])

            validate("named arguments",
                     self.kwargs,
                     inputs.kwargs,
                     are_named_args=True)

        def _forEachMatched(self, data, condition, doOnTrue, conditionMatches):
            if isinstance(data, (tuple, list)):
                return type(data)(self._forEachMatched(
                    d, condition, doOnTrue, conditionMatches) for d in data)
            if isinstance(data, dict):
                return {
                    key: self._forEachMatched(value, condition, doOnTrue,
                                              conditionMatches)
                    for key, value in data.items()
                }
            if condition(data):
                conditionMatches[0] = True
                return doOnTrue(data)
            return data

        def forEachMatchedAtLeastOnce(self, condition, doOnTrue=None):
            matches = [False]
            self._args = self._forEachMatched(self._args, condition, doOnTrue,
                                              matches)
            self._kwargs = self._forEachMatched(self._kwargs, condition,
                                                doOnTrue, matches)
            return matches[0]

        def forEach(self, fn):
            self._args = self._forEach(self._args, fn)
            self._kwargs = self._forEach(self._kwargs, fn)

        def asPackedFlatTuple(self, canonical_args=None):
            # Remove all the non torch.tensor types and flatten
            # any data structure.
            cargs = None if canonical_args is None else canonical_args.args
            ckwargs = None if canonical_args is None else canonical_args.kwargs
            return tuple(
                _utils.flattenTensorStructure(self._args, cargs) +
                _utils.flattenTensorStructure(self._kwargs, ckwargs))

    def __init__(self, model: Any):
        # Combine args and kwargs:
        if isinstance(model, _impl.OptimizerWrapper):
            sig = inspect.signature(model.model.forward)
        elif isinstance(model, torch.nn.Module):
            sig = inspect.signature(model.forward)
        elif callable(model):
            try:
                sig = inspect.signature(model)
            except ValueError:
                # ValueError: no signature found for builtin ...
                # If the callable is a Cython function then its signature
                # might not be available (E.g torch.nn.functional.logsigmoid)
                sig = None
        else:
            raise TypeError("Expected a torch.nn.Module or a callable")
        if sig is None:
            # If we couldn't extract the function's signature: be flexible
            # and default to "*args, **kwargs"
            self._varnames = ["args", "kwargs"]
            self._var_kinds = [
                inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
            ]
            self._defaults = {}
            self._has_variadic_arguments = True
        else:
            self._var_kinds = [p.kind for p in sig.parameters.values()]
            self._has_variadic_arguments = any(kind in [
                inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
            ] for kind in self._var_kinds)
            self._varnames = list(sig.parameters.keys())
            self._defaults = {
                name: p.default
                for name, p in sig.parameters.items()
                if p.default != inspect.Parameter.empty
            }

        self._warned_not_contiguous_input = False

    def __call__(self,
                 args: Any,
                 kwargs: Dict[str, Any],
                 fast_path: bool = False) -> Args:
        """Checks the inputs are of a supported type. Inputs must be
           tensors or tuples/lists of tensors. Will convert list to tuples
           as we can't natively support lists in the JIT.
        """
        in_tensors = ArgsParser.Args()
        assert self._has_variadic_arguments or len(args) + len(kwargs) <= len(
            self._varnames), ("Too many arguments provided: expected %s (%d) "
                              "but got %d") % (self._varnames,
                                               len(self._varnames),
                                               len(args) + len(kwargs))
        # Make sure all the arguments provided are allowed.
        if not self._has_variadic_arguments:
            for k in kwargs.keys():
                assert k in self._varnames, (
                    f"{k} is not a valid parameter."
                    f"Allowed values are {self._varnames}")

        variadic_pos_set = False
        for i, name in enumerate(self._varnames):
            is_variadic_pos = self._var_kinds[
                i] == inspect.Parameter.VAR_POSITIONAL
            is_variadic_keyword = self._var_kinds[
                i] == inspect.Parameter.VAR_KEYWORD

            if is_variadic_keyword:
                # A variadic keyword argument will consume all the remaining
                # kwargs
                used_names = self._varnames[:i]
                for k, v in kwargs.items():
                    if k not in used_names:
                        in_tensors.setNamedArg(k, v)
            elif i < len(args) or is_variadic_pos:
                # If it's a variadic parameter: consume all the remaining args
                # otherwise consume only one.
                if is_variadic_pos:
                    variadic_pos_set = True
                    a = args[i:]
                    # Clear args: all the arguments have been consumed
                    args = []
                else:
                    a = [args[i]]
                for idx, arg in enumerate(a):
                    if is_variadic_pos:
                        arg_name = f"*{name}[{idx}]"
                    else:
                        arg_name = name

                    # Non fast path for compilation, fast path for executing.
                    if not fast_path:
                        self._dictCheck(arg)

                    in_tensors.appendArg(arg, arg_name)

                assert name not in kwargs, ("Parameter %s was passed more "
                                            "than once") % name
            elif name in kwargs:
                # Non fast path for compilation, fast path for executing.
                if not fast_path:
                    self._dictCheck(kwargs[name])

                # Everything after a variadic positional argument must be named
                if variadic_pos_set:
                    in_tensors.setNamedArg(name, kwargs[name])
                else:
                    in_tensors.appendArg(kwargs[name], name)
            else:
                if name not in self._defaults:
                    raise _impl.createPoptorchError("Mandatory parameter "
                                                    f"{name} missing")
                value = self._defaults[name]
                # Everything after a variadic positional argument must be named
                if variadic_pos_set:
                    in_tensors.setNamedArg(name, value)
                else:
                    in_tensors.appendArg(value, name)

        if in_tensors.forEachMatchedAtLeastOnce(
                condition=lambda t: isinstance(t, torch.Tensor
                                               ) and not t.is_contiguous(),
                doOnTrue=lambda t: t.contiguous()):
            if not self._warned_not_contiguous_input:
                logger.warning("At least one input tensor is not contiguous: "
                               "non-contiguous tensors will be converted.")
                self._warned_not_contiguous_input = True

        return in_tensors

    def _dictCheck(self, data):
        work = [data]
        while len(work) > 0:
            d = work.pop()
            if isinstance(d, (tuple, list)):
                work.extend(d)
            elif isinstance(d, dict):
                logger.warning("Dicts as inputs only have partial support, "
                               "they can be accessed using literal keys, but "
                               "full Python functionality is not enabled. "
                               "Consider changing dict inputs to tuple.")
                return
