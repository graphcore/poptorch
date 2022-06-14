# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import torch
from poptorch import *  # pylint: disable=wildcard-import
from . import poptorch_core

# Hack so that print() works: Torch's print does a lot of slicing
# and select on the tensor before moving it to the CPU.
# This is annoying because it pollutes the graph, and generates some dynamic
# slices, etc.
# So, instead, we move the tensor to the CPU first, then we let torch do its
# thing.
# Upstream fix: https://github.com/pytorch/pytorch/pull/79287/files
real_str = torch._tensor_str._str  # pylint: disable=protected-access


def _str(tensor):
    if tensor.device.type == "xla":
        return real_str(tensor.to("cpu"))
    return real_str(tensor)


torch._tensor_str._str = _str  # pylint: disable=protected-access

poptorch_core.enableEagerMode()
