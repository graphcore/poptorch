#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import gc
import pytest


@pytest.fixture(autouse=True)
def cleanup():
    # Explicitly clean up to make sure we detach from the IPU and free the graph
    # before the next test starts.
    gc.collect()


# Source: https://raphael.codes/blog/customizing-your-pytest-test-suite-part-2/
def pytest_collection_modifyitems(session, config, items):  # pylint: disable=unused-argument
    if not config.getoption("hw_tests_only"):
        return
    # if --hw-tests-only is set: only keep tests with a "skipif" marker.
    # That's because all HW tests are protected by a:
    # @pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
    #                reason="Hardware IPU needed to test this feature")
    selected = []
    deselected = []
    for item in items:
        if any(item.iter_markers("skipif")):
            selected.append(item)
        else:
            deselected.append(item)
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def pytest_addoption(parser):
    parser.addoption("--hw-tests-only",
                     action="store_true",
                     default=False,
                     help="Only run HW tests")
