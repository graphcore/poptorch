# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import dataclasses
import torch

import helpers
from torch_geometric.data import Batch, Data

from gnn.nn.nn_utils import op_harness


def assert_(native_out, poptorch_out):
    def check_inner_field(x, y):
        assert isinstance(x, type(y)), \
            f"x type={type(x)} is different than y type={type(y)}"
        if isinstance(x, torch.Tensor):
            helpers.assert_allclose(actual=x,
                                    expected=y,
                                    atol=1e-04,
                                    rtol=1e-04,
                                    equal_nan=True)
        elif isinstance(x, (list, tuple)):
            for t, ct in zip(x, y):
                check_inner_field(t, ct)
        elif isinstance(x, (Batch, Data)):
            assert x.keys == y.keys, "Objects have different keys."
            for k in x.keys:
                check_inner_field(x[k], y[k])
        elif dataclasses.is_dataclass(x):
            for att in dir(x):
                x_field = getattr(x, att, None)
                if not callable(x_field) and isinstance(x_field, torch.Tensor):
                    check_inner_field(x_field, getattr(y, att, None))
        elif x is not None:
            assert False, f"Unsupported types: x type={type(x)}, y type=" \
                f"{type(y)}"

    check_inner_field(native_out, poptorch_out)


def pool_harness(op, inputs, assert_func=None):

    if assert_func is None:
        assert_func = assert_
    poptorch_out = op_harness(op, inputs, assert_func)

    return poptorch_out
