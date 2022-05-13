#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import enum
import gc
import pytest
import torch
import poptorch


@pytest.fixture(autouse=True)
def cleanup():
    # Explicitly clean up to make sure we detach from the IPU and free the graph
    # before the next test starts.
    gc.collect()


# Documentation about markers: https://docs.pytest.org/en/6.2.x/example/markers.html

mlir_available = poptorch.hasMlirSupportOnPlatform()
hw_available = poptorch.ipuHardwareIsAvailable()


def pytest_make_parametrize_id(val, argname):
    if val is None or isinstance(
            val, (bool, int, str, float, torch.dtype, enum.Enum)):
        return f"{argname}:{val}"
    if isinstance(val, type):
        return f"{argname}:{val.__name__}"

    # Use default
    return None


def pytest_configure(config):
    config.addinivalue_line(
        "markers", ("mlirSupportRequired: require MLIR to be available "
                    "on the platform"))
    config.addinivalue_line("markers",
                            ("ipuHardwareRequired: require IPU hardware to be "
                             "available on the platform"))


def pytest_runtest_setup(item):
    if any(item.iter_markers("mlirSupportRequired")):
        if not mlir_available:
            pytest.skip("No MLIR support on this platform.")

    if any(item.iter_markers("ipuHardwareRequired")):
        if not hw_available:
            pytest.skip("Hardware IPU needed to test this feature.")


# Source: https://raphael.codes/blog/customizing-your-pytest-test-suite-part-2/
def pytest_collection_modifyitems(session, config, items):  # pylint: disable=unused-argument
    if not config.getoption("hw_tests_only") and not config.getoption(
            "no_hw_tests"):
        return
    # if --hw-tests-only is set: only keep tests with a "ipuHardwareRequired"
    # marker.
    # if --no-hw-tests is set: keep only the other ones.
    hw_required = []
    hw_not_required = []
    for item in items:
        if any(item.iter_markers("ipuHardwareRequired")):
            hw_required.append(item)
        else:
            hw_not_required.append(item)
    if config.getoption("hw_tests_only"):
        config.hook.pytest_deselected(items=hw_not_required)
        items[:] = hw_required
    else:
        config.hook.pytest_deselected(items=hw_required)
        items[:] = hw_not_required


def pytest_addoption(parser):
    parser.addoption("--hw-tests-only",
                     action="store_true",
                     default=False,
                     help="Only run HW tests")
    parser.addoption("--no-hw-tests",
                     action="store_true",
                     default=False,
                     help="Exclude all tests requiring HW")
