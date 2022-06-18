#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
import pytest

import poptorch
import helpers


def index_op0(t, idx, v=None):
    if v is None:
        return t[idx]
    t[idx] = v
    return t


def index_op1(t, idx, v=None):
    if v is None:
        return t[idx, idx]
    t[idx, idx] = v
    return t


def index_op2(t, idx, v=None):
    if v is None:
        return t[:, idx]
    t[:, idx] = v
    return t


def index_op3(t, idx, v=None):
    if v is None:
        return t[idx, :, idx]
    t[idx, :, idx] = v
    return t


def index_op4(t, idx, v=None):
    if v is None:
        return t[:, :, idx]
    t[:, :, idx] = v
    return t


def index_op5(t, idx, v=None):
    if v is None:
        return t[:, idx, idx]
    t[:, idx, idx] = v
    return t


def index_op6(t, idx, v=None):
    if v is None:
        return t[idx, idx, idx, idx]
    t[idx, idx, idx, idx] = v
    return t


def index_op7(t, idx, v=None):
    if v is None:
        return t[:, :, :, idx]
    t[:, :, :, idx] = v
    return t


def index_op8(t, idx, v=None):
    if v is None:
        return t[:, idx, :, idx]
    t[:, idx, :, idx] = v
    return t


def index_harness(trace_model, op, idxs, is_index_put, v=None, is_mask=False):
    torch.manual_seed(42)
    t = torch.randn(2, 3, 4, 5)
    if not is_mask:
        idxs_tensors = [torch.tensor([i]) for i in idxs]
    else:
        idxs_tensors = [idxs]
    model = helpers.ModelWithWeights(op, t.shape)
    # The LR should be large enough to guarantee weights change
    optim = torch.optim.AdamW(model.parameters(), lr=0.1)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model,
                                            options=options,
                                            optimizer=optim)

    if is_index_put:
        if v is None:
            v = torch.zeros_like(op(t, *idxs_tensors))
        # Clone the tensor so that the original is unchanged by the in-place op
        native_out, _ = model((t.clone(), *idxs_tensors, v))
        poptorch_out, _ = poptorch_model((t, *idxs_tensors, v))
    else:
        native_out, _ = model((t, *idxs_tensors))
        poptorch_out, _ = poptorch_model((t, *idxs_tensors))

    # Inference test - check outputs
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


index_ops = [
    index_op0,
    index_op1,
    index_op2,
    index_op3,
    index_op4,
    index_op5,
    index_op6,
    index_op7,
    index_op8,
]

index_indices = ([[0]], [[1]], [[0, 1]], [[1, 0]], [[[0, 1], [1, 0]]])


@pytest.mark.parametrize("idxs", index_indices)
@pytest.mark.parametrize("op", index_ops)
@pytest.mark.parametrize("trace_model", [True, False])
def test_index(op, idxs, trace_model):
    if not trace_model:
        pytest.skip("TODO(T51159): Unsupported value kind: GenericList")
    index_harness(trace_model, op, idxs, False)


@pytest.mark.parametrize("trace_model", [True, False])
def test_index_on_max_indices(trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::max.dim")

    def op(x):
        _, argmax_tensor = torch.max(x, dim=1)
        b = x[:, argmax_tensor]
        return b, argmax_tensor

    inp_tensor = torch.rand(1, 10, 2)

    model = helpers.ModelWithWeights(op, inp_tensor.shape, lambda x: x[0])
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=options)

    native_out, _ = model((inp_tensor, ))
    poptorch_out, _ = poptorch_model((inp_tensor, ))

    # Inference test - check outputs
    for native, pop in zip(native_out, poptorch_out):
        helpers.assert_allclose(actual=pop, expected=native)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


@pytest.mark.parametrize("idxs", index_indices)
@pytest.mark.parametrize("op", index_ops)
@pytest.mark.parametrize("trace_model", [True, False])
def test_index_put(op, idxs, trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): RuntimeError: a leaf Variable that requires grad "
            "is being used in an in-place operation.")
    index_harness(trace_model, op, idxs, True)


@pytest.mark.parametrize("trace_model", [True, False])
def test_index_put_scalar(trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): RuntimeError: a leaf Variable that requires grad "
            "is being used in an in-place operation.")

    def op(t, idx, v):
        t[idx, idx] = v.item()
        return t

    # For each element e in t[0, 0], e = 0
    index_harness(trace_model, op, [[0]], True, torch.tensor([0]))


@pytest.mark.parametrize("trace_model", [True, False])
def test_index_put_broadcastable(trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): RuntimeError: a leaf Variable that requires grad "
            "is being used in an in-place operation.")
    v = torch.zeros(5)
    # For each row r in t[0, 0], r = [0, 0, 0, 0, 0]
    index_harness(trace_model, index_op1, [[0]], True, v)


@pytest.mark.parametrize("mask_size, dtype", [
    (1, torch.bool),
    (2, torch.uint8),
    (3, torch.bool),
    (4, torch.uint8),
])
@pytest.mark.parametrize("trace_model", [True, False])
def test_index_put_masked_fill(mask_size, dtype, trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): RuntimeError: a leaf Variable that requires grad "
            "is being used in an in-place operation.")
    torch.manual_seed(42)
    mask_shape = [2, 3, 4, 5][:mask_size]
    mask = (torch.rand(mask_shape) > 0.5).type(dtype)
    v = torch.tensor([0.])
    index_harness(trace_model, index_op0, mask, True, v=v, is_mask=True)


@pytest.mark.parametrize("mask_size, dtype", [
    (1, torch.bool),
    (2, torch.uint8),
    (3, torch.bool),
    (4, torch.uint8),
])
@pytest.mark.parametrize("trace_model", [True, False])
def test_index_put_masked_assign(mask_size, dtype, trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): RuntimeError: a leaf Variable that requires grad "
            "is being used in an in-place operation.")
    torch.manual_seed(42)
    mask_shape = [2, 3, 4, 5][:mask_size]
    mask = (torch.rand(mask_shape) > 0.5).type(dtype)
    v = torch.zeros([2, 3, 4, 5][mask_size:], dtype=torch.float32)
    if len(v.size()) == 0:
        # To avoid a size 0 tensor
        v = v.unsqueeze(0)
    index_harness(trace_model, index_op0, mask, True, v=v, is_mask=True)


@pytest.mark.parametrize("dim", range(-3, 3))
@pytest.mark.parametrize("trace_model", [True, False])
def test_index_select(dim, trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): Cannot index outside the tensor (dims 3) with "
            "dim (-1)")
    op = lambda src, index: src.index_select(dim, index)

    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    sz = x.shape[dim]
    indices = torch.randint(sz, (sz, ))

    model = helpers.ModelWithWeights(op, x.shape)
    native_out, _ = model((x, indices))

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=options)
    poptorch_out, _ = poptorch_model((x, indices))

    # Inference test - check outputs
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()
