# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import copy
from . import poptorch_core
from . import _options_impl


def enableEagerMode(*, headless: bool = False):
    """
    Enable eager mode returning the options class associated with this instance
    of eager mode

    When headless is true don't do any allocations or computation. This is
    useful for testing the dispatcher without running anything.
    """
    eager_options = poptorch_core.enableEagerMode(headless)
    eager_options.source_location_excludes = copy.copy(
        _options_impl.default_source_location_excludes)

    return eager_options


def markStep():
    """ Break the current lazy tensor trace and start executing it
     asynchronously. This doesn't do anything when not in eager mode or when
     use_lazy_tensor is off
    """
    poptorch_core.markStep()
