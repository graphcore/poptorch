# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import math
import torch

import helpers
from torch_geometric.data import Batch, Data
from torch_geometric.nn.pool.base import PoolingOutput
import poptorch

# Need to import poptorch_geometric to ensure that our arg parser implementation is
# registered with poptorch ahead of running these tests
import poptorch_geometric  # pylint: disable=unused-import


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
        elif isinstance(x, PoolingOutput):
            for att in dir(x):
                x_field = getattr(x, att, None)
                if not callable(x_field) and isinstance(x_field, torch.Tensor):
                    check_inner_field(x_field, getattr(y, att, None))
        elif x is not None:
            assert False, f"Unsupported types: x type={type(x)}, y type=" \
                f"{type(y)}"

    check_inner_field(native_out, poptorch_out)


def op_harness(op, inputs, assert_func=None):
    class ModelWW(helpers.ModelWithWeights):
        def __init__(self, op, first_input_shape):
            super().__init__(op, first_input_shape)
            self.op = op
            self.loss_fn = torch.nn.MSELoss()
            self.first_input_shape = first_input_shape
            self.first_input_numel = first_input_shape.numel()
            self.out_fn = torch.nn.Linear(self.first_input_numel,
                                          self.first_input_numel)
            self._weights_before = self.out_fn.weight.detach().clone()

        def forward(self, xs):
            if callable(getattr(op, "forward", None)):
                x = op.forward(*xs)
                l = 0
            else:
                x = self.op(*xs)
                if isinstance(x, (Batch, Data)):
                    x1 = torch.flatten(x.x)
                elif isinstance(x, tuple):
                    x1 = torch.flatten(x[0])
                else:
                    x1 = torch.flatten(x)
                if x1.shape.numel() != self.first_input_numel:
                    ratio = math.ceil(self.first_input_numel /
                                      x1.shape.numel())
                    x1 = x1.repeat(ratio)[:self.first_input_numel]
                if x1.dtype != torch.float:
                    x1 = x1.float()
                x1 = x1 if self.out_fn is None else self.out_fn(x1)
                x1 = x1.reshape(self.first_input_shape)
                target = torch.ones_like(x1)
                l = self.loss_fn(x1, target)
            return x, l

    if isinstance(inputs[0], (Batch, Data)):
        first_input_shape = inputs[0].x.shape
    else:
        first_input_shape = inputs[0].shape

    model = ModelWW(op, first_input_shape)

    # Run on CPU.
    native_out, _ = model(tuple(inputs))

    # The LR should be large enough that a single training step will
    # definitely cause weights to change
    optim = torch.optim.AdamW(model.parameters(), lr=0.1)

    # Run on IPU.
    poptorch_model = poptorch.trainingModel(model, optimizer=optim)
    poptorch_out, _ = poptorch_model(tuple(inputs))

    # Training test - check weights have changed
    poptorch_model.assert_weights_changed()

    if assert_func is None:
        assert_func = assert_
    assert_func(native_out, poptorch_out)

    return poptorch_out
