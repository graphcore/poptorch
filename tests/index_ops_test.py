#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import pytest

import poptorch


class IndexModel0(nn.Module):
    def forward(self, t, idx, v=None):
        if v is None:
            return t[idx]
        t += 0  # Ensure input is not modified in place
        t[idx] = v
        return t


class IndexModel1(nn.Module):
    def forward(self, t, idx, v=None):
        if v is None:
            return t[idx, idx]
        t += 0  # Ensure input is not modified in place
        t[idx, idx] = v
        return t


class IndexModel2(nn.Module):
    def forward(self, t, idx, v=None):
        if v is None:
            return t[:, idx]
        t += 0  # Ensure input is not modified in place
        t[:, idx] = v
        return t


class IndexModel3(nn.Module):
    def forward(self, t, idx, v=None):
        if v is None:
            return t[idx, :, idx]
        t += 0  # Ensure input is not modified in place
        t[idx, :, idx] = v
        return t


class IndexModel4(nn.Module):
    def forward(self, t, idx, v=None):
        if v is None:
            return t[:, :, idx]
        t += 0  # Ensure input is not modified in place
        t[:, :, idx] = v
        return t


class IndexModel5(nn.Module):
    def forward(self, t, idx, v=None):
        if v is None:
            return t[:, idx, idx]
        t += 0  # Ensure input is not modified in place
        t[:, idx, idx] = v
        return t


class IndexModel6(nn.Module):
    def forward(self, t, idx, v=None):
        if v is None:
            return t[idx, idx, idx, idx]
        t += 0  # Ensure input is not modified in place
        t[idx, idx, idx, idx] = v
        return t


class IndexModel7(nn.Module):
    def forward(self, t, idx, v=None):
        if v is None:
            return t[:, :, :, idx]
        t += 0  # Ensure input is not modified in place
        t[:, :, :, idx] = v
        return t


class IndexModel8(nn.Module):
    def forward(self, t, idx, v=None):
        if v is None:
            return t[:, idx, :, idx]
        t += 0  # Ensure input is not modified in place
        t[:, idx, :, idx] = v
        return t


def index_harness(model, idxs, pass_value, v=None):
    t = torch.arange(120, dtype=torch.int32).reshape(2, 3, 4, 5)
    idxs_tensors = []
    for i in idxs:
        idxs_tensors.append(torch.tensor(i))
    idxs_tuple = tuple(idxs_tensors)
    poptorch_model = poptorch.inferenceModel(model)

    if pass_value:
        if v is None:
            v = torch.empty_like(model(t, *idxs_tuple),
                                 dtype=torch.int32).fill_(2 * 3 * 4 * 5)
        native_out = model(t, *idxs_tuple, v)
        poptorch_out = poptorch_model(t, *idxs_tuple, v)
    else:
        native_out = model(t, *idxs_tuple)
        poptorch_out = poptorch_model(t, *idxs_tuple)

    assert native_out.size() == poptorch_out.size()
    torch.testing.assert_allclose(poptorch_out, native_out)


index_models = (
    IndexModel0(),
    IndexModel1(),
    IndexModel2(),
    IndexModel3(),
    IndexModel4(),
    IndexModel5(),
    IndexModel6(),
    IndexModel7(),
    IndexModel8(),
)

index_indices = ([[0]], [[1]], [[0, 1]], [[1, 0]], [[[0, 1], [1, 0]]])


@pytest.mark.parametrize("idxs", index_indices)
@pytest.mark.parametrize("model", index_models)
def test_index(model, idxs):
    index_harness(model, idxs, False)


@pytest.mark.parametrize("idxs", index_indices)
@pytest.mark.parametrize("model", index_models)
def test_index_put(model, idxs):
    index_harness(model, idxs, True)


def test_index_put_assign_scalar():
    class Model(nn.Module):
        def forward(self, t, idx, v):
            t += 0  # Ensure input is not modified in place
            t[:, idx] = v.item()
            return t

    # For each element e in t[:, 0], e = 120
    index_harness(Model(), [[0]], True, torch.tensor([2 * 3 * 4 * 5]))


def test_index_put_assign_broadcastable():
    v = torch.empty(5, dtype=torch.int32).fill_(2 * 3 * 4 * 5)
    # For each row r in t[:, 0], r = [120, 120, 120, 120, 120]
    index_harness(IndexModel2(), [[0]], True, v)
