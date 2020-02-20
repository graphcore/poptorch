import os
import sys

# Don't think we should keep this. It's done in popart, but we should just be able to add this path to LD_LIBRARY_PATH in build/activate.sh?
lp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
lp = os.path.abspath(lp)
sys.path.insert(0, lp)

import torch
import poptorch_core
from poptorch_core import *

pipeline_stage = torch.ops.poptorch.pipeline_stage
virtual_graph = torch.ops.poptorch.virtual_graph

# From pytorch/torch/jit/__init__.py
def make_tuple(example_inputs):
    if isinstance(example_inputs, (torch.Tensor, dict)):
        return (example_inputs,)
    # done primarily so that weird iterables fail here and not pybind11 code
    if not isinstance(example_inputs, tuple):
        return tuple(example_inputs)
    return example_inputs


def traceAndRun(path, tensors):
    in_tensors = make_tuple(tensors)
    return poptorch_core.traceAndRun(path, in_tensors)
