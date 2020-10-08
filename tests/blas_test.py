#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import poptorch


def blas_op(op, input1, input2, out):
    class Model(torch.nn.Module):
        def __init__(self, op):
            super(Model, self).__init__()
            self.op = op

        def forward(self, x, y, out=None):
            return self.op(x, y, out=out)

    model = Model(op)
    args = [input1, input2]
    if out is not None:
        args.append(out)
    # Run on CPU.
    nativeOut = model(*args)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(*args)

    torch.testing.assert_allclose(nativeOut,
                                  poptorch_out,
                                  atol=1e-05,
                                  rtol=1e-05,
                                  equal_nan=True)
    if out is not None:
        torch.testing.assert_allclose(nativeOut,
                                      out,
                                      atol=1e-05,
                                      rtol=1e-05,
                                      equal_nan=True)


@pytest.mark.parametrize("optional_out", [True, False])
def test_matmul(optional_out):
    torch.manual_seed(42)

    input1 = torch.randn([10, 200])
    input2 = torch.randn([200, 45])
    out = torch.randn([10, 45]) if optional_out else None

    blas_op(torch.matmul, input1, input2, out)


@pytest.mark.parametrize("optional_out", [True, False])
def test_bmm(optional_out):
    input1 = torch.randn([12, 10, 200])
    input2 = torch.randn([12, 200, 33])
    out = torch.randn([12, 10, 33]) if optional_out else None

    blas_op(torch.bmm, input1, input2, out)
