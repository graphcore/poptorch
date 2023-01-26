#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import random
import enum
import os.path as osp
import gc
import pytest
import torch
import helpers
import numpy as np
import poptorch


@pytest.fixture(autouse=True)
def cleanup():
    # Explicitly clean up to make sure we detach from the IPU and free the graph
    # before the next test starts.
    gc.collect()


# Documentation about markers: https://docs.pytest.org/en/6.2.x/example/markers.html

hw_available = poptorch.ipuHardwareIsAvailable()


def pytest_make_parametrize_id(val, argname):
    if isinstance(val, enum.Enum):
        return f"{argname}:{val.name}"
    if val is None or isinstance(val, (bool, int, str, float, torch.dtype)):
        return f"{argname}:{val}"
    if isinstance(val, type):
        return f"{argname}:{val.__name__}"

    # Use default
    return None


def pytest_configure(config):
    config.addinivalue_line("markers",
                            ("ipuHardwareRequired: require IPU hardware to be "
                             "available on the platform"))
    config.addinivalue_line("markers",
                            ("excludeFromReducedTesting: exclude from "
                             "reduced testing runs"))
    config.addinivalue_line("markers",
                            ("extendedTestingOnly: to only include "
                             "in extended testing runs because it takes a "
                             "long time to run"))
    if config.getoption("collectonly"):
        helpers.is_running_tests = False
    helpers.running_reduced_testing = config.getoption("reduced_testing")


def pytest_runtest_setup(item):
    # Is it a test with parameters?
    if hasattr(item, 'callspec'):
        # Does it have a trace_model parameter ?
        trace_model = item.callspec.params.get("trace_model")
        if trace_model is not None:
            if trace_model:
                pytest.skip("Tracing is no longer supported: skipping.")

    if any(item.iter_markers("ipuHardwareRequired")):
        if not hw_available:
            pytest.skip("Hardware IPU needed to test this feature.")


# Source: https://raphael.codes/blog/customizing-your-pytest-test-suite-part-2/
def pytest_collection_modifyitems(session, config, items):  # pylint: disable=unused-argument
    # if --extended-tests is set: include all the tests with a
    # "extendedTestingOnly" marker (Even if --hw-tests-only is set).
    # if --hw-tests-only is set: only keep tests with a "ipuHardwareRequired"
    # marker.
    # if --no-hw-tests is set: keep only the other ones.
    hw_required = []
    hw_not_required = []
    force_include = []
    force_exclude = []
    include_extended = config.getoption("extended_tests")
    for item in items:
        if helpers.running_reduced_testing and any(
                item.iter_markers("excludeFromReducedTesting")):
            force_exclude.append(item)
        elif any(item.iter_markers("extendedTestingOnly")):
            if include_extended:
                force_include.append(item)
            else:
                force_exclude.append(item)
        elif any(item.iter_markers("ipuHardwareRequired")):
            hw_required.append(item)
        else:
            hw_not_required.append(item)
    if config.getoption("hw_tests_only"):
        config.hook.pytest_deselected(items=hw_not_required + force_exclude)
        items[:] = hw_required + force_include
    elif config.getoption("no_hw_tests"):
        config.hook.pytest_deselected(items=hw_required + force_exclude)
        items[:] = hw_not_required + force_include
    else:
        config.hook.pytest_deselected(items=force_exclude)
        items[:] = hw_required + hw_not_required + force_include


def pytest_sessionfinish(session, exitstatus):
    # Exit status 5 means no tests were collected -> this is not an error.
    # In our case this is not an error because some files might only contain
    # HW tests for example and therefore won't have any test to run if
    # --hw-tests-only is used.
    if exitstatus == 5:
        session.exitstatus = 0


def pytest_addoption(parser):
    parser.addoption("--hw-tests-only",
                     action="store_true",
                     default=False,
                     help="Only run HW tests")
    parser.addoption("--no-hw-tests",
                     action="store_true",
                     default=False,
                     help="Exclude all tests requiring HW")
    parser.addoption("--extended-tests",
                     action="store_true",
                     default=False,
                     help=("Include all tests marked with "
                           "'extendedTestingOnly' (Takes precedence over"
                           " --no-hw-tests)"))
    parser.addoption("--reduced-testing",
                     action="store_true",
                     default=False,
                     help=("Run some tests with a reduced "
                           "number of parameters"))
    parser.addoption("--seed",
                     type=int,
                     default=0,
                     help=("Set the seed for running the tests."))
    parser.addoption("--external-datasets-dir",
                     type=str,
                     default=osp.join(osp.dirname(osp.abspath(__file__)),
                                      ".datasets"),
                     help=("The directory where the external datasets will be "
                           "downloaded."))


@pytest.fixture(autouse=True, scope="function")
def random_seed(pytestconfig):
    """Set the random seed for all tests in this directory. autouse=True will
    use this fixture in every test. Seed can be overridden with --seed on the
    command line to alter the seed for testing purposes. By default uses 0 for
    all tests.
    """
    seed = 0
    if hasattr(pytestconfig, "seed"):
        seed = pytestconfig.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
