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


def index_harness(op, idxs, pass_value, v=None):
    torch.manual_seed(42)
    t = torch.randn(2, 3, 4, 5)
    idxs_tensors = [torch.tensor(i) for i in idxs]
    model = helpers.ModelWithWeights(op, t.shape)
    # The LR should be large enough to guarantee weights change
    optim = torch.optim.AdamW(model.parameters(), lr=0.1)
    poptorch_model = poptorch.trainingModel(model, optimizer=optim)

    if pass_value:
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
# TODO(T40635): Remove this list and use index_ops in test_index_put when
# the index_put bug is fixed
index_put_ops = [
    index_op0,
    index_op1,
    index_op6,
]

index_indices = ([[0]], [[1]], [[0, 1]], [[1, 0]], [[[0, 1], [1, 0]]])


@pytest.mark.parametrize("idxs", index_indices)
@pytest.mark.parametrize("op", index_ops)
def test_index(op, idxs):
    index_harness(op, idxs, False)


def test_index_on_max_indices():
    def op(x):
        _, argmax_tensor = torch.max(x, dim=1)
        b = x[:, argmax_tensor]
        return b, argmax_tensor

    inp_tensor = torch.rand(1, 10, 2)

    model = helpers.ModelWithWeights(op, inp_tensor.shape, lambda x: x[0])
    poptorch_model = poptorch.trainingModel(model)

    native_out, _ = model((inp_tensor, ))
    poptorch_out, _ = poptorch_model((inp_tensor, ))

    # Inference test - check outputs
    for native, pop in zip(native_out, poptorch_out):
        helpers.assert_allclose(actual=pop, expected=native)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


@pytest.mark.parametrize("idxs", index_indices)
@pytest.mark.parametrize("op", index_put_ops)
def test_index_put(op, idxs):
    index_harness(op, idxs, True)


def test_index_put_scalar():
    def op(t, idx, v):
        t[idx, idx] = v.item()
        return t

    # For each element e in t[0, 0], e = 0
    index_harness(op, [[0]], True, torch.tensor([0]))


def test_index_put_broadcastable():
    v = torch.zeros(5)
    # For each row r in t[0, 0], r = [0, 0, 0, 0, 0]
    index_harness(index_op1, [[0]], True, v)


@pytest.mark.parametrize("dim", range(-3, 3))
def test_index_select(dim):
    op = lambda src, index: src.index_select(dim, index)

    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    sz = x.shape[dim]
    indices = torch.randint(sz, (sz, ))

    model = helpers.ModelWithWeights(op, x.shape)
    native_out, _ = model((x, indices))

    poptorch_model = poptorch.trainingModel(model)
    poptorch_out, _ = poptorch_model((x, indices))

    # Inference test - check outputs
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()
