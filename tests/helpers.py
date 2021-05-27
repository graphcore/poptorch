# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import copy
import functools
import re
from abc import ABC
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


def disableAllModels():
    return {"POPTORCH_IPU_MODEL": "0", "POPTORCH_SMALL_IPU_MODEL": "0"}


def propagateInputShapes(graph, dummyInputs):
    for graphInput, dummyInput in zip(graph.inputs(), dummyInputs):
        graphInput.inferTypeFrom(dummyInput)
    poptorch_core.propagateInputShapes(graph)


def trainingModelWithLoss(model, loss, options=None, optimizer=None):
    class TrainingModelWithLoss(torch.nn.Module):
        def __init__(self, model, loss):
            super().__init__()
            self._real_call = model.__call__
            # The original model *must* be stored in the wrapper
            # even if it's not used (The tracer will inspect it
            # for parameters).
            self._model = model
            self._loss = loss

        def forward(self, args, loss_inputs):  # pylint: disable=unused-argument
            assert False, ("Shouldn't be called, signature should match"
                           " the one of __call__")

        def __call__(self, args, loss_inputs):
            output = self._real_call(args)
            loss = self._loss(output, loss_inputs)
            return output, loss

    # Create a copy of the original model in case it needs to be wrapped
    maybe_wrapped_model = copy.copy(model)

    # Store the real __call__ method before PoplarExecutor wraps it
    training_model = TrainingModelWithLoss(maybe_wrapped_model, loss)
    return poptorch.PoplarExecutor(model=training_model,
                                   options=options,
                                   training=True,
                                   optimizer=optimizer,
                                   user_model=model,
                                   poptorch_version=poptorch.__version__)


# Wrapper model with weights to test that gradients are generated
# and updated in a graph with a given op - Linear layer added to
# ensure some weights exist
#
# This is an abstract class and the forward function must be implemented
# in a subclass
class AbstractModelWithWeights(torch.nn.Module, ABC):
    def __init__(self, op, input_shapes, out_fn=None, loss_fn=None):
        super().__init__()
        self.op = op
        self.input_shapes = input_shapes
        numel = sum([shape.numel() for shape in input_shapes])
        self.lin = torch.nn.Linear(numel, numel)
        # Copy original weights for training test
        self._weights_before = self.lin.weight.detach().clone()
        # A function of the output that returns what the backwards pass should
        # propagate through. For example, torch.median returns values and indices
        # but the loss should only be calculated using the values. If unspecified,
        # defaults to an identity function
        self.out_fn = out_fn
        # If the loss fn takes a target param, this value must be included in a
        # wrapper function that only takes a single input before passing to this
        # constructor
        self.loss_fn = loss_fn if not loss_fn is None \
            else lambda x: poptorch.identity_loss(x**2, reduction='sum')

    def handle_output(self, x):
        loss_in = x if self.out_fn is None else self.out_fn(x)
        if isinstance(loss_in, tuple):
            l = self.loss_fn(*loss_in)
        else:
            l = self.loss_fn(loss_in)
        return x, l

    def assert_weights_changed(self):
        weights_after = self.lin.weight.detach().clone()
        assert not torch.allclose(self._weights_before, weights_after)


class UnaryModelWithWeights(AbstractModelWithWeights):
    def __init__(self, op, input_shape, out_fn=None, loss_fn=None):
        super().__init__(op, [input_shape], out_fn, loss_fn)

    # Flatten input, pass through linear layer of same size
    # and pass output to the op
    def forward(self, x):
        x = torch.flatten(x)
        x = self.lin(x)
        x = x.reshape(self.input_shapes[0])
        x = self.op(x)
        return self.handle_output(x)


class BinaryModelWithWeights(AbstractModelWithWeights):
    def __init__(self,
                 op,
                 input_shape1,
                 input_shape2,
                 out_fn=None,
                 loss_fn=None):
        super().__init__(op, [input_shape1, input_shape2], out_fn, loss_fn)

    # Flatten both inputs, concatenate into a a single 1-dim tensor
    # and pass through linear layer of same size, then split and assign
    # the output according to the sizes of the inputs and pass to the op
    def forward(self, x1, x2):
        x1_flat = torch.flatten(x1)
        x2_flat = torch.flatten(x2)
        x = torch.cat((x1_flat, x2_flat))
        x = self.lin(x)
        x1_out = x[:self.input_shapes[0].numel()]
        x2_out = x[self.input_shapes[0].numel():]
        x1_out = x1_out.reshape(self.input_shapes[0])
        x2_out = x2_out.reshape(self.input_shapes[1])
        x = self.op(x1_out, x2_out)
        return self.handle_output(x)


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
    def __init__(self, capfd):
        out, err = capfd.readouterr()
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

    def assert_matches(self, *exprs):
        """Assert there is a line in the log matching all the regular expressions provided
        """
        for line in self._lines:
            if all([re.search(e, line) for e in exprs]):
                # Found a line matching all the exprs
                return
        raise ValueError(
            f"{self._log}"
            "\n No line in the above log matches all of the expressions "
            f"{exprs}")

    def assert_no_matches(self, *exprs):
        """Assert there is no line matching all the regular expressions provided"""
        for line in self._lines:
            if all([re.search(e, line) for e in exprs]):
                # Found a line matching all the exprs
                raise ValueError(
                    f"{line}"
                    "\n The line above matches all of the expressions "
                    f"{exprs}")
