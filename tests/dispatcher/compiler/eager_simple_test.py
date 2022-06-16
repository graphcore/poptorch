#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
import torchvision.models as models
import pytest
import helpers


def simple_add(capfd):
    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel

    capfd.readouterr()  # Clear the log

    def fn(input, check_log=False):
        x = input + 5
        if check_log:
            # Check the add was lowered and executed.
            log = helpers.LogChecker(capfd)
            log.assert_not_contains("CPU -> IPU")
            log.assert_not_contains("IPU -> CPU")
            log.assert_contains("Graph lowered to popit")
        return x * 3

    input = torch.ones([10])
    cpu = fn(input)
    log = helpers.LogChecker(capfd)
    log.assert_isEmpty()
    input = input.to("xla")
    log = helpers.LogChecker(capfd)
    log.assert_contains("CPU -> IPU")
    ipu = fn(input, check_log=True)
    # Check the multiplication was also lowered and executed.
    log = helpers.LogChecker(capfd)
    log.assert_not_contains("CPU -> IPU")
    log.assert_not_contains("IPU -> CPU")
    log.assert_contains("Graph lowered to popit")
    print(f"Result cpu: {cpu} ipu: {ipu}")
    # Check the print triggered a copy to host
    log = helpers.LogChecker(capfd)
    log.assert_not_contains("CPU -> IPU")
    log.assert_contains("IPU -> CPU")
    helpers.assert_allclose(expected=cpu, actual=ipu.cpu())
    # Check .cpu() triggered a copy to host
    log = helpers.LogChecker(capfd)
    log.assert_not_contains("CPU -> IPU")
    log.assert_contains("IPU -> CPU")


@pytest.mark.mlirSupportRequired
@pytest.mark.ipuHardwareRequired
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
def test_simple_add_hw(capfd):
    simple_add(capfd)


@pytest.mark.mlirSupportRequired
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
def test_simple_add(capfd):
    pytest.skip("PopIT doesn't currently support IPUModel")
    simple_add(capfd)


@pytest.mark.ipuHardwareRequired
@pytest.mark.mlirSupportRequired
def test_squeezenet():
    pytest.skip("Takes too long to run for now")
    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel

    input = torch.randn([1, 3, 224, 224])

    model = models.squeezenet1_1(pretrained=False)
    model.eval()

    cpu = model(input)

    model.to("xla")
    input = input.to("xla")

    ipu = model(input)

    print(f"Result cpu: {cpu} ipu: {ipu}")
    helpers.assert_allclose(expected=cpu, actual=ipu.cpu())


@pytest.mark.ipuHardwareRequired
@pytest.mark.mlirSupportRequired
def test_resnet18():
    pytest.skip("TODO(T64252): Failing to compile & run the norm layer")

    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel

    input = torch.randn([1, 3, 224, 224])

    model = models.resnet18(pretrained=False)
    model.eval()

    cpu = model(input)

    model.to("xla")
    input = input.to("xla")

    ipu = model(input)

    print(f"Result cpu: {cpu} ipu: {ipu}")
    helpers.assert_allclose(expected=cpu, actual=ipu.cpu())
