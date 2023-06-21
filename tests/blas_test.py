#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import helpers
import poptorch


def blas_op(op, input1, input2, out, atol=1e-04, rtol=1e-04):
    class Model(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, x, y, out=None):
            return self.op(x, y, out=out)

    model = Model(op)
    args = [input1, input2]
    if out is not None:
        args.append(out)

    native_out = None
    # Matmul fp16 is not supported on the CPU
    if input1.dtype != torch.half and input2.dtype != torch.half:
        # Run on CPU.
        native_out = model(*args)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(*args)

    if native_out is not None:
        helpers.assert_allclose(expected=native_out,
                                actual=poptorch_out,
                                atol=atol,
                                rtol=rtol,
                                equal_nan=True)
    if out is not None and native_out is not None:
        helpers.assert_allclose(expected=native_out,
                                actual=out,
                                atol=atol,
                                rtol=rtol,
                                equal_nan=True)


@pytest.mark.parametrize("out", [True, False])
@pytest.mark.parametrize("shapes", [([10, 200], [200, 45], [10, 45]),
                                    ([10, 200], [200], [10]),
                                    ([200], [200, 45], [1, 45]),
                                    ([200], [200], [])])
def test_matmul(out, shapes):
    torch.manual_seed(42)

    if len(shapes[0]) == 1 and len(shapes[1]) == 1 and out:
        pytest.skip(
            "TODO(T71439) No shape inference handler for aten::fill_.Tensor")

    input1 = torch.randn(shapes[0])
    input2 = torch.randn(shapes[1])
    out = torch.randn(shapes[2]) if out else None

    blas_op(torch.matmul, input1, input2, out)


@pytest.mark.parametrize("mode",
                         (poptorch.MatMulSerializationMode.InputChannels,
                          poptorch.MatMulSerializationMode.ReducingDim,
                          poptorch.MatMulSerializationMode.OutputChannels,
                          poptorch.MatMulSerializationMode.Disabled))
@pytest.mark.parametrize("factor", (2, 5, 10))
@pytest.mark.parametrize("keep_precision", [True, False])
def test_serializedMatMul(mode, factor, keep_precision):
    torch.manual_seed(42)

    input1 = torch.rand(1, 10, 200)

    input2_dim = 45

    if mode == poptorch.MatMulSerializationMode.OutputChannels:
        # Ensure the value is a multiple of factor
        input2_dim = input2_dim // factor * factor

    input2 = torch.rand(200, input2_dim)

    def serialise_matmal_op(input, other, out):
        assert out is None
        return poptorch.serializedMatMul(input, other, mode, factor,
                                         keep_precision)

    if keep_precision:
        input1 = input1.half()
        input2 = input2.half()
        blas_op(serialise_matmal_op,
                input1,
                input2,
                None,
                rtol=0.01,
                atol=0.05)
    else:
        blas_op(serialise_matmal_op, input1, input2, None)


@pytest.mark.parametrize("optional_out", [True, False])
def test_bmm(optional_out):
    input1 = torch.randn([12, 10, 200])
    input2 = torch.randn([12, 200, 33])
    out = torch.randn([12, 10, 33]) if optional_out else None

    blas_op(torch.bmm, input1, input2, out)


@pytest.mark.parametrize("bias", [True, False])
def test_matmul_training(bias):
    N, M, K, C = 100, 9, 7, 5

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(42)
            self.linear = torch.nn.Linear(K, K, bias=bias)
            self.softmax = torch.nn.LogSoftmax(dim=1)
            self.loss = torch.nn.L1Loss(reduction="mean")

        def forward(self, x, y, target):
            x = self.linear(x)
            x = torch.matmul(x, y)
            return x, self.loss(x, target)

    torch.manual_seed(42)
    model = Net()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    torch.manual_seed(42)
    poptorch_model = poptorch.trainingModel(model, optimizer=optimizer)
    x = torch.randn(N, M, K)
    y = torch.randn(K, K)
    target = torch.empty(N, M, K, dtype=torch.long).random_(0, C)

    for _ in range(0, 400):
        optimizer.zero_grad()
        poptorch_out, poptorch_loss = poptorch_model(x, y, target)
        native_out, native_loss = model(x, y, target)
        native_loss.backward(retain_graph=True)
        optimizer.step()

    helpers.assert_allclose(actual=poptorch_out,
                            expected=native_out,
                            rtol=1e-02,
                            atol=1e-02)
    helpers.assert_allclose(actual=poptorch_loss,
                            expected=native_loss,
                            rtol=1e-03,
                            atol=1e-03)


@pytest.mark.parametrize(
    "params",
    [
        # input_shape, beta, alpha
        ((3, 7), 1.0, 1.0),
        ((3, 1), 1.0, 0.75),
        ((1, 7), 0.75, 1.0),
        ((1), 0.75, 0.75),
    ])
def test_addmm(params):
    torch.manual_seed(42)

    input_shape, beta, alpha = params

    t1 = torch.randn(input_shape)
    t2 = torch.randn(3, 5)
    t3 = torch.randn(5, 7)

    class AddmmModel(torch.nn.Module):
        def __init__(self, beta, alpha):
            super().__init__()
            self.beta = beta
            self.alpha = alpha

        def forward(self, x1, x2, x3):
            return torch.addmm(x1, x2, x3, beta=self.beta, alpha=self.alpha)

    model = AddmmModel(beta, alpha)
    cpu_result = model(t1, t2, t3)
    ipu_result = poptorch.inferenceModel(model)(t1, t2, t3)

    helpers.assert_allclose(expected=cpu_result, actual=ipu_result)


@pytest.mark.parametrize(
    "params",
    [
        # input_shape, beta, alpha
        ((3, 7), 1.0, 1.0),
        ((3, 1), 1.0, 0.75),
        ((1, 7), 0.75, 1.0),
        ((1), 0.75, 0.75),
    ])
def test_baddbmm(params):
    torch.manual_seed(42)

    input_shape, beta, alpha = params

    t1 = torch.randn(input_shape)
    t2 = torch.randn(2, 3, 5)
    t3 = torch.randn(2, 5, 7)

    class AddmmModel(torch.nn.Module):
        def __init__(self, beta, alpha):
            super().__init__()
            self.beta = beta
            self.alpha = alpha

        def forward(self, x1, x2, x3):
            return torch.baddbmm(x1, x2, x3, beta=self.beta, alpha=self.alpha)

    model = AddmmModel(beta, alpha)
    cpu_result = model(t1, t2, t3)
    ipu_result = poptorch.inferenceModel(model)(t1, t2, t3)

    helpers.assert_allclose(expected=cpu_result, actual=ipu_result)


@pytest.mark.parametrize("input_shape", [(20, 10)])
@pytest.mark.parametrize("beta", [0, .5])
@pytest.mark.parametrize("alpha", [0, 1.5])
@pytest.mark.parametrize("use_out", [True, False])
def test_addmv(input_shape, beta, alpha, use_out):
    torch.manual_seed(42)

    mat = torch.randn(input_shape)
    vec = torch.randn(input_shape[1])
    inp = torch.randn(input_shape[0])

    if beta == 0:
        # NaNs in input should be ignored
        inp[0] = float('nan')
    if alpha == 0:
        # NaNs in vec or mat should be ignored
        mat[0, 0] = float('nan')
        vec[0] = float('nan')

    output = torch.empty(input_shape[0]) if use_out else None

    class AddmvModel(torch.nn.Module):
        def __init__(self, beta, alpha):
            super().__init__()
            self.beta = beta
            self.alpha = alpha

        def forward(self, inp, mat, vec, out=None):
            result = torch.addmv(inp,
                                 mat,
                                 vec,
                                 beta=self.beta,
                                 alpha=self.alpha,
                                 out=out)
            if self.beta == 0 and self.alpha == 0:
                # Avoid empty compute graph
                result += torch.zeros_like(inp)
            return result

    model = AddmvModel(beta, alpha)
    cpu_result = model(inp, mat, vec, out=output)
    ipu_result = poptorch.inferenceModel(model)(inp, mat, vec, output)

    helpers.assert_allclose(expected=cpu_result, actual=ipu_result)
    if use_out is True:
        helpers.assert_allclose(expected=cpu_result, actual=output)
