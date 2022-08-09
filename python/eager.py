# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import copy
from poptorch import *  # pylint: disable=wildcard-import
from . import poptorch_core
from . import _options_impl

eager_options = poptorch_core.enableEagerMode()
eager_options.source_location_excludes = copy.copy(
    _options_impl.default_source_location_excludes)
