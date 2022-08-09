# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import copy
from poptorch import *  # pylint: disable=wildcard-import
from . import poptorch_core
from . import _options_impl

eager_options = poptorch_core.enableEagerMode()
eager_options.source_location_excludes = copy.copy(
    _options_impl.default_source_location_excludes)


def markStep():
    """ Break the current lazy tensor trace and start executing it
     asynchronously. This doesn't do anything when not in eager mode or when
     use_lazy_tensor is off
    """
    poptorch_core.markStep()
