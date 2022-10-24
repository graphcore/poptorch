#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import inspect
import copy
import pytest
import torch
import torch.nn as nn
import helpers
from poptorch.experimental import IPUContext, IPUScope
import poptorch


def test_simple_test():
    input = torch.ones([10])

    with IPUScope([input]) as ipu:
        x = input.to("xla")
        x = x + 5
        x = x * 3
        ipu.outputs(x)

    helpers.assert_allequal(actual=ipu(input),
                            expected=torch.empty(10).fill_(18.0))


def test_simple_conv():
    input = torch.ones([1, 5, 25, 25])

    conv = nn.Conv2d(5, 10, 5)
    cpu_conv = copy.deepcopy(conv)

    with IPUScope([input], model=conv) as ipu:
        x = input.to("xla")
        x = conv(x)
        ipu.outputs(x)

    cpu = cpu_conv(input)
    ipu = ipu(input)

    helpers.assert_allclose(expected=cpu,
                            actual=ipu,
                            atol=1e-05,
                            rtol=1e-05,
                            equal_nan=True)


def test_tensor_constant():
    def f(x):
        return x + torch.tensor([1.0, 2.0, 3.0], device=helpers.outputDevice())

    input = torch.rand(3)
    with IPUScope([input]) as ipu:
        y = input.to("xla")
        y = f(y)
        ipu.outputs(y)

    cpu = f(input)
    ipu = ipu(input)

    helpers.assert_allequal(expected=cpu, actual=ipu)


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
@pytest.mark.parametrize("mode", ["default", "show_all", "hide_all"])
@pytest.mark.parametrize("compiler",
                         [poptorch.Compiler.PopART, poptorch.Compiler.MLIR])
def test_source_location(capfd, compiler, mode):
    layer = torch.nn.Linear(1, 2)
    expected_filename = inspect.stack()[0].filename
    # +3 -> We expect to see f()'s return line in the log
    expected_line = inspect.stack()[0].lineno + 3

    def f(x):
        return layer(x)

    opts = None
    if mode == "show_all":
        opts = poptorch.Options()
        # Clear the list: show everything
        opts.sourceLocationExcludes([])
    elif mode == "hide_all":
        opts = poptorch.Options()
        # All paths have a '/' in them so we essentially exclude everything.
        opts.appendToLocationExcludes("/")

    input = torch.Tensor([[1.], [-1.]])
    ipu = IPUContext(f, model=layer, options=opts, compiler=compiler)

    ipu(input)
    log = helpers.LogChecker(capfd)
    if compiler == poptorch.Compiler.PopART:
        if mode == "show_all":
            # If we clear the list of exclusions we will point at Torch's internals
            log.assert_matches(
                "site-packages/torch/nn/functional.py.* was lowered to")
            log.assert_no_matches(
                f"{expected_filename}:{expected_line}.* was lowered to")
        elif mode == "hide_all":
            log.assert_contains(") was lowered to")  # no filename
            log.assert_no_matches(
                "site-packages/torch/nn/functional.py.* was lowered to")
            log.assert_no_matches(
                f"{expected_filename}:{expected_line}.* was lowered to")
        else:
            # By default: we point at the user code
            log.assert_no_matches(
                "site-packages/torch/nn/functional.py.* was lowered to")
            log.assert_matches(
                f"{expected_filename}:{expected_line}.* was lowered to")
    else:
        if mode == "show_all":
            # If we clear the list of exclusions we will point at Torch's internals
            log.assert_matches(
                "poptorch.addmm.*site-packages/torch/nn/functional.py")
            log.assert_no_matches(
                f"poptorch.addmm.*{expected_filename}:{expected_line}")
        elif mode == "hide_all":
            log.assert_matches(r"poptorch.addmm.*\[unknown\]")  # no filename
            log.assert_no_matches(
                "poptorch.addmm.*site-packages/torch/nn/functional.py")
            log.assert_no_matches(
                f"poptorch.addmm.*{expected_filename}:{expected_line}")
        else:
            # By default: we point at the user code
            log.assert_no_matches(
                "poptorch.addmm.*site-packages/torch/nn/functional.py")
            log.assert_matches(
                f"poptorch.addmm.*{expected_filename}:{expected_line}")
