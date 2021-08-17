# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import functools
import os
import re
import torch
import poptorch
import poptorch.poptorch_core as poptorch_core  # type: ignore


def assert_allclose(*, actual=None, expected=None, **kwargs):
    """Assertion function that enforces passing the 'actual' and 'expected'
    arguments to torch.testing.assert_allclose in the correct order by forcing
    the use of keyword arguments. This improves error reporting in case of
    assertion failures.

    :param actual: torch.Tensor, array-like of torch.Tensor objects or
           scalar value that is tested.
    :param expected: torch.Tensor, array-like of torch.Tensor objects or
           scalar value that is used as a reference.
    :param kwargs: kwargs passed to torch.testing.assert_allclose.
    """
    assert actual is not None and expected is not None, (
        "'actual' and 'expected' keyword arguments must be present")

    in_types = (type(actual), type(expected))
    if in_types == (torch.Tensor, torch.Tensor):
        assert actual.shape == expected.shape, (
            "Shape of 'actual' (%s) should be the same as shape of"
            " 'expected' (%s)") % (actual.shape, expected.shape)
    elif in_types in ((list, list), (tuple, tuple)):
        assert len(actual) == len(expected), (
            "Length of 'actual' (%s) should be the same as length of"
            " 'expected' (%s)") % (len(actual), len(expected))
        for a, e in zip(actual, expected):
            assert_allclose(actual=a, expected=e, **kwargs)
        return

    torch.testing.assert_allclose(actual, expected, **kwargs)


def assert_allequal(*, actual=None, expected=None, msg=''):
    """Assertion function that enforces passing the 'actual' and 'expected'
    arguments to torch.testing.assert_allclose in the correct order by forcing
    the use of keyword arguments. This improves error reporting in case of
    assertion failures. Additionally, rtol=0 and atol=0 are passed to
    torch.testing.assert_allclose as this results in identity comparison for
    integer and boolean tensors.

    :param actual: torch.Tensor or scalar value that is tested.
    :param expected: torch.Tensor or scalar value that is used as a reference.
    :param msg: message passed to torch.testing.assert_allclose.
    """
    assert actual is not None and expected is not None, (
        "'actual' and 'expected' keyword arguments must be present")

    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        assert actual.shape == expected.shape, (
            "Shape of 'actual' (%s) should be the same as shape of"
            " 'expected' (%s)") % (actual.shape, expected.shape)

    torch.testing.assert_allclose(actual, expected, rtol=0, atol=0, msg=msg)


def disableSmallModel():
    # POPTORCH_IPU_MODEL takes precedence over POPTORCH_SMALL_IPU_MODEL
    if not poptorch.ipuHardwareIsAvailable():
        return {"POPTORCH_IPU_MODEL": "1"}
    return {}


def forceSmallModel():
    # POPTORCH_IPU_MODEL takes precedence over POPTORCH_SMALL_IPU_MODEL
    return {"POPTORCH_IPU_MODEL": "0", "POPTORCH_SMALL_IPU_MODEL": "1"}


def disableAllModels():
    return {"POPTORCH_IPU_MODEL": "0", "POPTORCH_SMALL_IPU_MODEL": "0"}


def propagateInputShapes(graph, dummyInputs):
    for graphInput, dummyInput in zip(graph.inputs(), dummyInputs):
        graphInput.inferTypeFrom(dummyInput)
    poptorch_core.propagateInputShapes(graph)


# Wrapper model with weights to test that gradients are generated
# and updated in a graph with a given op - Linear layer added to
# ensure some weights exist
class ModelWithWeights(torch.nn.Module):
    def __init__(self, op, first_input_shape, out_fn=None, loss_fn=None):
        super().__init__()
        self.op = op
        numel = first_input_shape.numel()
        self.first_input_shape = first_input_shape
        self.lin = torch.nn.Linear(numel, numel)
        # Copy original weights for training test
        self._weights_before = self.lin.weight.detach().clone()
        # A function of the output that returns what the backwards pass should
        # propagate through. For example, torch.median returns values and indices
        # but the loss should only be calculated using the values. If unspecified,
        # defaults to an identity function
        self.out_fn = out_fn
        # If the loss fn takes more than 1 param (e.g. a target), these extra params
        # must be wrapped in a function that only takes a single input
        self.loss_fn = loss_fn if not loss_fn is None \
            else lambda x: poptorch.identity_loss(x**2, reduction='sum')

    # Flatten first input, pass through linear layer of same size
    # and pass reassembled inputs to op
    def forward(self, xs):
        assert isinstance(xs, tuple)
        x1 = torch.flatten(xs[0])
        x1 = self.lin(x1)
        x1 = x1.reshape(self.first_input_shape)
        x = self.op(x1, *xs[1:])
        loss_in = x if self.out_fn is None else self.out_fn(x)
        if isinstance(loss_in, tuple):
            l = self.loss_fn(*loss_in)
        else:
            l = self.loss_fn(loss_in)
        return x, l

    def assert_weights_changed(self):
        weights_after = self.lin.weight.detach().clone()
        assert not torch.allclose(self._weights_before, weights_after)


class PrintCapfdOnExit:
    """Helper that prints the content of capfd on exit

    Useful if a test fails before its output validation step."""

    def __init__(self, capfd):
        self.capfd = capfd

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        out, err = self.capfd.readouterr()
        log = out + err
        with self.capfd.disabled():
            if log:
                print(log)


def printCapfdOnExit(func):
    """Decorator to print the content of capfd after the wrapped function
    exits."""

    @functools.wraps(func)
    def wrapper(capfd, *args, **kwargs):
        with PrintCapfdOnExit(capfd):
            func(*args, **kwargs, capfd=capfd)

    return wrapper


def overridePoptorchLogLevel(level=None):
    """Decorator to override the PopTorch log level for the duration of the test"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if level is not None:
                poptorch.setLogLevel(level)
            func(*args, **kwargs)
            poptorch.setLogLevel(os.environ.get("POPTORCH_LOG_LEVEL", "WARN"))

        return wrapper

    return decorator


def overridePopartLogLevel(level=None):
    """Decorator to override the Popart log level for the duration of the test"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if level is not None:
                poptorch._logging.setPopartLogLevel(level)  # pylint: disable=protected-access
            func(*args, **kwargs)
            poptorch._logging.setPopartLogLevel(  # pylint: disable=protected-access
                os.environ.get("POPART_LOG_LEVEL", "WARN"))

        return wrapper

    return decorator


class LogIterator:
    def __init__(self, lines):
        self._lines = lines
        self._current = 0
        self._num_lines = len(lines)
        self._all_checks = []

    def findNext(self, *exprs):
        """Find the next line in the log matching all the regular expressions provided"""
        self._all_checks.append(exprs)
        while True:
            assert self._current < self._num_lines, (
                "\n".join(self._lines) +
                "\n The log above doesn't contain lines matching all "
                "these expressions:\n  " +
                "\n  ".join(str(e) for e in self._all_checks))
            line = self._lines[self._current]
            self._current += 1
            if all([re.search(e, line) for e in exprs]):
                return line

    def findAll(self, expr):
        """Return all lines in the log matching the provided regular expression"""
        matching_lines = []
        for line in self._lines:
            match = re.search(expr, line)
            if match is not None:
                matching_lines.append(match)
        return matching_lines


class LogChecker:
    def __init__(self, capfd_or_str):
        if isinstance(capfd_or_str, str):
            self._log = capfd_or_str
        else:
            out, err = capfd_or_str.readouterr()
            self._log = out + err
        self._lines = self._log.split('\n')

    def createIterator(self):
        return LogIterator(self._lines)

    def assert_contains(self, *strings):
        """Assert there is a line in the log matching all the strings provided
        """
        if len(strings) == 1:
            assert strings[0] in self._log, (f"{self._log}"
                                             "\ndoes not contain "
                                             f"'{strings[0]}'")
        else:
            assert any([
                all([s in line for s in strings]) for line in self._lines
            ]), (f"{self._log}"
                 "\n No line in the above log contains all of the strings "
                 f"{strings}")

    def assert_not_contains(self, *strings):
        """Assert there is no line in the log matching all the strings provided
        """
        if len(strings) == 1:
            assert strings[0] not in self._log, (f"{self._log}"
                                                 "\ncontains "
                                                 f"'{strings[0]}'")
        else:
            for line in self._lines:
                if all([s in line for s in strings]):
                    # Found a line matching all the strings
                    raise ValueError(
                        f"{line}"
                        "\n The line above matches all of the strings "
                        f"{strings}")

    def _string_matches_exprs(self, s, exprs):
        return all([re.search(e, s) for e in exprs])

    def assert_matches(self, *exprs, per_line=True):
        """Assert the log matches all the regular expressions provided
        """
        if per_line:
            # Found a line matching all the exprs
            if any([
                    self._string_matches_exprs(line, exprs)
                    for line in self._lines
            ]):
                return
        else:
            # Search the entire log at once
            if self._string_matches_exprs(self._log, exprs):
                return

        any_line_in = "any line in " if per_line else ""
        raise ValueError(
            f"{self._log}"
            f"\n All of the expressions do not match {any_line_in}"
            f"the log {exprs}")

    def assert_no_matches(self, *exprs, per_line=True):
        """Assert the log does not match all the regular expressions provided"""
        if per_line:
            for line in self._lines:
                if self._string_matches_exprs(line, exprs):
                    # Found a line matching all the exprs
                    raise ValueError(
                        f"{line}"
                        "\n The line above matches all of the expressions "
                        f"{exprs}")
        else:
            if self._string_matches_exprs(self._log, exprs):
                # The log matches all the exprs
                raise ValueError(
                    f"{self._log}"
                    "\n The log above matches all of the expressions "
                    f"{exprs}")
