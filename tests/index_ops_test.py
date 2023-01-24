#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
import pytest

import helpers
import poptorch


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


def index_harness(op, idx, is_index_put, v=None, is_mask=False):
    torch.manual_seed(42)
    t = torch.randn(2, 3, 4, 5)
    if not is_mask:
        idx_tensor = torch.tensor(idx)
    else:
        idx_tensor = idx
    model = helpers.ModelWithWeights(op, t.shape)
    # The LR should be large enough to guarantee weights change
    optim = torch.optim.AdamW(model.parameters(), lr=0.1)
    poptorch_model = poptorch.trainingModel(model, optimizer=optim)

    if is_index_put:
        if v is None:
            v = torch.zeros_like(op(t, idx_tensor))
        # Clone the tensor so that the original is unchanged by the in-place op
        native_out, _ = model((t.clone(), idx_tensor, v))
        poptorch_out, _ = poptorch_model((t, idx_tensor, v))
    else:
        native_out, _ = model((t, idx_tensor))
        poptorch_out, _ = poptorch_model((t, idx_tensor))

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

index_indices = ([0], [[1]], [0, 1], [[1, 0]], [[0, 1], [1, 0]])


@pytest.mark.parametrize("idxs", index_indices)
@pytest.mark.parametrize("op", index_ops)
def test_index(op, idxs):
    index_harness(op, idxs, False)


def test_index_bool_mask_failure():
    with pytest.raises(
            poptorch.poptorch_core.Error,
            match=r"Operations using aten::nonzero are unsupported "
            r"because the output shape is determined by the tensor "
            r"values\. The IPU cannot support dynamic output shapes\."):
        index_harness(index_ops[0], [True, False], False)


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
@pytest.mark.parametrize("op", index_ops)
def test_index_put(op, idxs):
    index_harness(op, idxs, True)


def test_index_put_scalar():
    def op(t, idx, v):
        t[idx, idx] = v
        return t

    # For each element e in t[0, 0], e = 0
    index_harness(op, [[0]], True, 0)


def test_index_put_broadcastable():
    v = torch.zeros(5)
    # For each row r in t[0, 0], r = [0, 0, 0, 0, 0]
    index_harness(index_op1, [[0]], True, v)


@pytest.mark.parametrize("mask_size, dtype", [
    (1, torch.bool),
    (2, torch.uint8),
    (3, torch.bool),
    (4, torch.uint8),
])
def test_index_put_masked_fill(mask_size, dtype):
    torch.manual_seed(42)
    mask_shape = [2, 3, 4, 5][:mask_size]
    mask = (torch.rand(mask_shape) > 0.5).type(dtype)
    v = torch.tensor([0.])
    index_harness(index_op0, mask, True, v=v, is_mask=True)


@pytest.mark.parametrize("mask_size, dtype", [
    (1, torch.bool),
    (2, torch.uint8),
    (3, torch.bool),
    (4, torch.uint8),
])
def test_index_put_masked_assign(mask_size, dtype):
    torch.manual_seed(42)
    mask_shape = [2, 3, 4, 5][:mask_size]
    mask = (torch.rand(mask_shape) > 0.5).type(dtype)
    v = torch.zeros([2, 3, 4, 5][mask_size:], dtype=torch.float32)
    if len(v.size()) == 0:
        # To avoid a size 0 tensor
        v = v.unsqueeze(0)
    index_harness(index_op0, mask, True, v=v, is_mask=True)


def get_index_fill_fn(dim):
    def index_fill(t, idx, v):
        t.index_fill_(dim, idx, v)
        return t

    return index_fill


@pytest.mark.parametrize("value", (-1, torch.tensor(-1)))
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_index_fill(value, dim):
    torch.manual_seed(42)
    op = get_index_fill_fn(dim)
    index_harness(op, [0, 2], True, value)


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


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
@pytest.mark.parametrize("dim", [0, 1])
def test_vectorized_scatter(capfd, dim):
    def op(out, index, src):
        if dim == 0:
            out[index, :] = src
        else:
            out[:, index] = src

        return out

    torch.manual_seed(0)
    N = 20
    out = torch.randn(N, 30)
    sz = out.shape[dim] - N // 10
    indices = torch.randint(sz, (sz, ))
    src_sz = (sz, out.shape[1]) if dim == 0 else (out.shape[0], sz)
    src = torch.randn(src_sz)

    model = helpers.ModelWithWeights(op, out.shape)
    # Clone the tensor so that the original is unchanged by the in-place op
    native_out, _ = model((out.clone(), indices, src))

    poptorch_model = poptorch.trainingModel(model)
    poptorch_out, _ = poptorch_model((out.clone(), indices, src))

    # Inference test - check outputs
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()

    it = helpers.LogChecker(capfd).createIterator()
    it.findNext("Using vectorized ScatterReduce with none reduction")
