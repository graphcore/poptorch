# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import copy
import inspect
import torch

# Do not import any poptorch.* here: it will break the poptorch module
from . import _impl
from ._logging import logger
from . import _utils


class ArgsParser:
    class Args:
        def __init__(self, tracing, varnames):
            self.args = []
            self.first_none = None
            self._tracing = tracing
            self._varnames = varnames

        def clone(self):
            clone = ArgsParser.Args(self._tracing, self._varnames)
            clone.args = copy.copy(self.args)
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
            if self._tracing and inputs.first_none != self.first_none:
                raise _impl.createPoptorchError(
                    "Number of arguments mismatch: "
                    f"{self.first_none} "
                    f"arguments used to compile the model and "
                    f"{inputs.first_none} provided this time")

            if len(inputs.args) != len(self.args):
                raise _impl.createPoptorchError(
                    "Number of arguments mismatch: expected "
                    f"{len(self.args)} arguments but got "
                    f"{len(inputs.args)}.{end}")

            def validate(name, compiled, input):
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
                    # Other types are compiled in the graph (scalars, etc) and
                    # therefore should be an exact match to the value used to
                    # compile the model.
                    if compiled != input:
                        raise _impl.createPoptorchError(
                            f"Value mismatch for {name}: "
                            f"expected {compiled} but got {input}.{end}")

            for i, arg in enumerate(self.args):
                validate(self._varnames[i], arg, inputs.args[i])

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
            self.args = self._forEachMatched(self.args, condition, doOnTrue,
                                             matches)
            return matches[0]

        def forEach(self, fn):
            self.args = self._forEach(self.args, fn)

        def asTuple(self):
            # Lists are hard to parse in the C++ because their size is not
            # known in the IValue, so convert them to tuples.
            def convert(input):
                if isinstance(input, (tuple, list)):
                    return tuple([convert(d) for d in input])
                return input

            return tuple([convert(a) for a in self.args])

        def asPackedFlatTuple(self):
            # Remove all the non torch.tensor types and flatten
            # any data structure.
            return tuple(_utils.flattenTensorStructure(self.args))

    def __init__(self, model, tracing=True):
        # Combine args and kwargs:
        if isinstance(model, _impl.OptimizerWrapper):
            sig = inspect.signature(model.model.forward)
        else:
            sig = inspect.signature(model.forward)

        self._has_variadic_arguments = any([
            p.kind in [p.VAR_POSITIONAL, p.VAR_KEYWORD]
            for p in sig.parameters.values()
        ])
        self._varnames = list(sig.parameters.keys())
        self._defaults = [p.default for p in sig.parameters.values()]
        self._warned_not_contiguous_input = False
        self._tracing = tracing

    def __call__(self, args, kwargs, fast_path=False):
        """Checks the inputs are of a supported type. Inputs must be
           tensors or tuples/lists of tensors. Will convert list to tuples
           as we can't natively support lists in the JIT.
        """
        in_tensors = ArgsParser.Args(self._tracing, self._varnames)
        assert self._has_variadic_arguments or len(args) + len(kwargs) <= len(
            self._varnames), ("Too many arguments provided: expected %s (%d) "
                              "but got %d") % (self._varnames,
                                               len(self._varnames),
                                               len(args) + len(kwargs))
        first_optional = len(self._varnames) - len(self._defaults)
        none_passed = []

        # Make sure all the arguments provided are allowed.
        for k in kwargs.keys():
            assert k in self._varnames, (
                f"{k} is not a valid parameter."
                f"Allowed values are {self._varnames}")

        for i, name in enumerate(self._varnames):
            if i < len(args):
                has_list = self._errorOnDictReturnTrueIfList(args[i], name, [])

                # Non fast path for compilation, fast path for executing.
                if not fast_path:
                    if has_list:
                        logger.warning(
                            "Lists as inputs only have partial support, they "
                            "can be accessed but full Python functionality is "
                            "not enabled. Consider changing input to tuple.")

                in_tensors.args.append(args[i])

                assert name not in kwargs, ("Parameter %s was passed more "
                                            "than once") % name
            elif name in kwargs:
                assert not none_passed, (
                    "Torch doesn't support passing tensors (%s)"
                    " after the following parameters have defaulted to None."
                    " %s") % (name, ", ".join(none_passed))
                has_list = self._errorOnDictReturnTrueIfList(
                    kwargs[name], name, [])

                # Non fast path for compilation, fast path for executing.
                if not fast_path:
                    if has_list:
                        logger.warning(
                            "Lists as inputs only have partial support, they "
                            "can be accessed but full Python functionality is "
                            "not enabled. Consider changing input to tuple.")
                in_tensors.args.append(kwargs[name])
            else:
                assert i >= first_optional, ("Mandatory parameter %s "
                                             "missing") % name
                value = self._defaults[i - first_optional]
                # We only need to keep track of None values when tracing because
                # torch.jit.trace() can't handle them.
                if value is None and self._tracing:
                    if in_tensors.first_none is None:
                        in_tensors.first_none = i
                    none_passed.append("%s (%d)" % (name, i))
                if not none_passed:
                    in_tensors.args.append(value)

        if in_tensors.forEachMatchedAtLeastOnce(
                condition=lambda t: isinstance(t, torch.Tensor
                                               ) and not t.is_contiguous(),
                doOnTrue=lambda t: t.contiguous()):
            if not self._warned_not_contiguous_input:
                logger.warning("At least one input tensor is not contiguous: "
                               "non-contiguous tensors will be converted.")
                self._warned_not_contiguous_input = True

        # The checks past this point are specific to torch.jit.trace() because
        # of the limited support it has for None and default values.
        if not self._tracing:
            return in_tensors

        if in_tensors.first_none is None:
            in_tensors.first_none = len(self._varnames)

        # filter-out trailing None arguments when they default to None
        # Extending this to any argument set to its default value has
        # proven problematic - the trace may be computed with fewer
        # inputs than intended.
        for i in reversed(range(len(in_tensors.args))):
            if in_tensors.args[i] is not None:
                break
            if self._defaults[i] is not None:
                break
            in_tensors.args.pop()
            if in_tensors.first_none == i:
                in_tensors.first_none = None

        # assert we are not passing None parameters to avoid a cryptic error
        assert None not in in_tensors.args, \
            "'None' may not be passed as explicit model argument. It may " + \
            "only be used as default initialiser"

        return in_tensors

    def _errorOnDictReturnTrueIfList(self, data, arg_name, stack_list):
        has_list = False
        if isinstance(data, (tuple, list)):
            for idx, d in enumerate(data):
                stack_list.append(idx)
                has_list &= self._errorOnDictReturnTrueIfList(
                    d, arg_name, stack_list)
                stack_list.pop()

            if isinstance(data, list):
                has_list = True

        if isinstance(data, dict):
            stack_list = [str(s) for s in stack_list]
            end_msg = arg_name
            if stack_list:
                end_msg += "[" + "][".join(stack_list) + "]"
            end_msg += " = " + str(data)

            raise TypeError(
                "Dictionaries are not supported as input arguments,"
                " including when nested in tuples.\nReceived dict " + end_msg)
        return has_list
