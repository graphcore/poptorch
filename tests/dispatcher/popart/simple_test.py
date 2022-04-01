#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import inspect
import pytest
import torch
import torch.nn as nn
from poptorch.experimental import IPUScope
import poptorch
import helpers


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_simple_test():
    input = torch.ones([10])

    with IPUScope([input]) as ipu:
        x = input + 5
        x = x * 3
        ipu.outputs(x)

    helpers.assert_allequal(actual=ipu(input),
                            expected=torch.empty(10).fill_(18.0))


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_simple_conv():
    input = torch.ones([1, 5, 25, 25])

    conv = nn.Conv2d(5, 10, 5)

    with IPUScope([input], conv.named_parameters()) as ipu:
        x = conv(input)
        ipu.outputs(x)

    cpu = conv(input)
    ipu = ipu(input)

    helpers.assert_allclose(expected=cpu,
                            actual=ipu,
                            atol=1e-05,
                            rtol=1e-05,
                            equal_nan=True)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_tensor_constant():
    def f(x):
        return x + torch.tensor([1.0, 2.0, 3.0])

    input = torch.rand(3)
    with IPUScope([input]) as ipu:
        y = f(input)
        ipu.outputs(y)

    cpu = f(input)
    ipu = ipu(input)

    helpers.assert_allequal(expected=cpu, actual=ipu)


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
@pytest.mark.parametrize("mode", ["default", "show_all", "hide_all"])
@pytest.mark.parametrize("compiler",
                         [poptorch.Compiler.PopART, poptorch.Compiler.MLIR])
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
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
    with IPUScope([input],
                  layer.named_parameters(),
                  compile_using=compiler,
                  options=opts) as ipu:
        y = f(input)
        ipu.outputs([y])

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
                "poptorch.transpose.*site-packages/torch/nn/functional.py")
            log.assert_no_matches(
                f"poptorch.transpose.*{expected_filename}:{expected_line}")
        elif mode == "hide_all":
            log.assert_matches(
                r"poptorch.transpose.*\[unknown\]")  # no filename
            log.assert_no_matches(
                "poptorch.transpose.*site-packages/torch/nn/functional.py")
            log.assert_no_matches(
                f"poptorch.transpose.*{expected_filename}:{expected_line}")
        else:
            # By default: we point at the user code
            log.assert_no_matches(
                "poptorch.transpose.*site-packages/torch/nn/functional.py")
            log.assert_matches(
                f"poptorch.transpose.*{expected_filename}:{expected_line}")
