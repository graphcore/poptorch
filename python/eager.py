# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from poptorch import *  # pylint: disable=wildcard-import
from . import _eager_helpers

eager_options = _eager_helpers.enableEagerMode()
