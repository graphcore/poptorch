#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import pytest

import poptorch


class IndexModel1(nn.Module):
    def forward(self, t, idx):
        return t[idx]


class IndexModel2(nn.Module):
    def forward(self, t, idx):
        return t[idx, idx]


class IndexModel3(nn.Module):
    def forward(self, t, idx):
        return t[:, idx]


class IndexModel4(nn.Module):
    def forward(self, t, idx):
        return t[idx, :, idx]


class IndexModel5(nn.Module):
    def forward(self, t, idx):
        return t[:, :, idx]


class IndexModel6(nn.Module):
    def forward(self, t, idx):
        return t[:, idx, idx]


class IndexModel7(nn.Module):
    def forward(self, t, idx):
        return t[:, idx, :, idx]


@pytest.mark.parametrize("idxs",
                         ([[0]], [[1]], [[0, 1]], [[1, 0]], [[[0, 1], [1, 0]]])
                         )
@pytest.mark.parametrize(
    "model", {
        IndexModel1(),
        IndexModel2(),
        IndexModel3(),
        IndexModel4(),
        IndexModel5(),
        IndexModel6(),
        IndexModel7()
    })
def test_index(model, idxs):
    t = torch.arange(120).reshape(2, 3, 4, 5)
    idxs_tensors = []
    for i in idxs:
        idxs_tensors.append(torch.tensor(i))
    idxs_tuple = tuple(idxs_tensors)
    poptorch_model = poptorch.inferenceModel(model)

    # Run on CPU
    native_out = model(t, *idxs_tuple)
    # Run on IPU
    poptorch_out = poptorch_model(t, *idxs_tuple)

    assert native_out.size() == poptorch_out.size()
    torch.testing.assert_allclose(poptorch_out, native_out)
