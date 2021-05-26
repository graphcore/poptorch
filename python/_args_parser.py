# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import copy
import inspect
import torch

# Do not import any poptorch.* here: it will break the poptorch module
from . import _impl
from ._logging import logger


class ArgsParser:
    class Args:
        def __init__(self):
            self._args = []
            self.first_none = None

        def clone(self):
            clone = ArgsParser.Args()
            clone._args = copy.copy(self._args)  # pylint: disable=protected-access
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
                conditionMatches.setTrue()
                return doOnTrue(data)
            return data

        def forEachMatchedAtLeastOnce(self, condition, doOnTrue=None):
            class ConditionMatches:
                def __init__(self):
                    self._matches = False

                def __bool__(self):
                    return self._matches

                def setTrue(self):
                    self._matches = True

            matches = ConditionMatches()
            self._args = self._forEachMatched(self._args, condition, doOnTrue,
                                              matches)
            return bool(matches)

        def forEach(self, fn):
            self._args = self._forEach(self._args, fn)

        def asTuple(self):
            return tuple(self._args)

    def __init__(self, model):
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

    def __call__(self, args, kwargs):
        """Calls the wrapped model with the given tensors. Inputs must be
        tensors or tuples/lists of tensors.
        Will compile for IPU on the first invocation.
        """
        in_tensors = ArgsParser.Args()
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
                self._errorOnListOrDict(args[i], name, [])
                in_tensors._args.append(args[i])
                assert name not in kwargs, ("Parameter %s was passed more "
                                            "than once") % name
            elif name in kwargs:
                assert not none_passed, (
                    "Torch doesn't support passing tensors (%s)"
                    " after the following parameters have defaulted to None."
                    " %s") % (name, ", ".join(none_passed))
                self._errorOnListOrDict(kwargs[name], name, [])
                in_tensors._args.append(kwargs[name])
            else:
                assert i >= first_optional, ("Mandatory parameter %s "
                                             "missing") % name
                value = self._defaults[i - first_optional]
                if value is None:
                    if in_tensors.first_none is None:
                        in_tensors.first_none = i
                    none_passed.append("%s (%d)" % (name, i))
                if not none_passed:
                    in_tensors._args.append(value)
        if in_tensors.first_none is None:
            in_tensors.first_none = len(self._varnames)

        # filter-out trailing None arguments when they default to None
        # Extending this to any argument set to its default value has
        # proven problematic - the trace may be computed with fewer
        # inputs than intended.
        for i in reversed(range(len(in_tensors._args))):
            if in_tensors._args[i] is not None:
                break
            if self._defaults[i] is not None:
                break
            in_tensors._args.pop()
            if in_tensors.first_none == i:
                in_tensors.first_none = None

        # assert we are not passing None parameters to avoid a cryptic error
        assert None not in in_tensors._args, \
            "'None' may not be passed as explicit model argument. It may " + \
            "only be used as default initialiser"

        if in_tensors.forEachMatchedAtLeastOnce(
                condition=lambda t: isinstance(t, torch.Tensor
                                               ) and not t.is_contiguous(),
                doOnTrue=lambda t: t.contiguous()):
            if not self._warned_not_contiguous_input:
                logger.warning("At least one input tensor is not contiguous: "
                               "non-contiguous tensors will be converted.")
                self._warned_not_contiguous_input = True

        return in_tensors

    def _errorOnListOrDict(self, data, arg_name, stack_list):
        if isinstance(data, (tuple)):
            for idx, d in enumerate(data):
                stack_list.append(idx)
                self._errorOnListOrDict(d, arg_name, stack_list)
                stack_list.pop()

        if isinstance(data, (dict, list)):
            stack_list = [str(s) for s in stack_list]
            end_msg = arg_name
            if stack_list:
                end_msg += "[" + "][".join(stack_list) + "]"
            end_msg += " = " + str(data)

        if isinstance(data, dict):
            raise TypeError(
                "Dictionaries are not supported as input arguments,"
                " including when nested in tuples.\nReceived dict " + end_msg)

        if isinstance(data, list):
            raise TypeError(
                "Lists are not supported as input arguments,"
                " including when nested in tuples.\nReceived list " + end_msg)
