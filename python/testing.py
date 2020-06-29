# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch


# Return true if both the structure and the content of ref and other match
def allclose(ref, other):
    if isinstance(ref, torch.Tensor):
        return torch.allclose(other, ref)
    if isinstance(ref, tuple):
        if not isinstance(other, tuple) or len(ref) != len(other):
            return False
    elif isinstance(ref, list):
        if not isinstance(other, list) or len(ref) != len(other):
            return False
    else:
        assert "%s not supported" % type(ref)
    return all([allclose(r, other[i]) for i, r in enumerate(ref)])
