# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import copy
import functools
import re
import torch
import poptorch
import poptorch.poptorch_core as poptorch_core


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

    training_model = copy.copy(model)

    maybe_wrapped_model = training_model

    if optimizer and optimizer.param_groups:
        maybe_wrapped_model = poptorch._impl.OptimizerWrapper(  # pylint: disable=protected-access
            training_model, optimizer)

    # Store the real __call__ method before PoplarExecutor wraps it
    return poptorch._impl.PoplarExecutor(  # pylint: disable=protected-access
        model=TrainingModelWithLoss(maybe_wrapped_model, loss),
        options=options,
        training=True,
        optimizer=optimizer,
        user_model=model)


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
