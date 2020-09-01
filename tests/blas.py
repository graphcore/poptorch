#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch


def blas_op(op, input1, input2, eq):
    class Model(torch.nn.Module):
        def __init__(self, op):
            super(Model, self).__init__()
            self.op = op

        def forward(self, x, y):
            return self.op(x, y)

    model = Model(op)

    # Run on CPU.
    nativeOut = model(input1, input2)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input1, input2)

    assert eq(nativeOut, poptorch_out)


def blas_op_optional_arg(op, input1, input2, out, eq):
    class Model(torch.nn.Module):
        def __init__(self, op):
            super(Model, self).__init__()
            self.op = op

        def forward(self, x, y, out):
            return self.op(x, y, out=out)

    model = Model(op)

    # Run on CPU.
    nativeOut = model(input1, input2, out)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input1, input2, out)

    assert eq(nativeOut, poptorch_out)
    assert eq(nativeOut, out)


def test_blas_ops_float():
    torch.manual_seed(42)

    input1 = torch.randn([10, 200])
    input2 = torch.randn([200, 45])
    out = torch.randn([10, 45])

    def compare(x, y):
        return torch.allclose(x, y, atol=1e-05, equal_nan=True)

    blas_op(torch.matmul, input1, input2, compare)
    blas_op_optional_arg(torch.matmul, input1, input2, out, compare)

    input1 = torch.randn([12, 10, 200])
    input2 = torch.randn([12, 200, 33])
    out = torch.randn([12, 10, 33])

    blas_op(torch.bmm, input1, input2, compare)
    blas_op_optional_arg(torch.bmm, input1, input2, out, compare)
