#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import helpers
import poptorch


class FiveAdder(torch.nn.Module):
    def forward(self, in_1, in_2, in_3, in_4, in_5):
        return in_1 + in_2 + in_3 + in_4 + in_5


def test_non_contiguous():
    torch.manual_seed(23148)

    model = FiveAdder()
    poptorch_model = poptorch.inferenceModel(model)

    OUTER_DIM = 1000
    INNER_DIM = 40

    nc1 = torch.randn([OUTER_DIM, INNER_DIM + 1])[:, 0:INNER_DIM]
    nc2 = torch.transpose(torch.randn([INNER_DIM, OUTER_DIM]), 0, 1)
    nc3 = torch.tensor([1.0]).expand([OUTER_DIM, INNER_DIM])

    c1 = torch.randn([OUTER_DIM, INNER_DIM])
    c2 = torch.randn([2, OUTER_DIM, INNER_DIM])[0, :, :]

    assert not nc1.is_contiguous()
    assert not nc2.is_contiguous()
    assert not nc3.is_contiguous()

    assert c1.is_contiguous()
    assert c2.is_contiguous()

    native_out = model(nc1, c1, nc2, c2, nc3)
    poptorch_out = poptorch_model(nc1, c1, nc2, c2, nc3)

    assert native_out.shape == (OUTER_DIM, INNER_DIM)

    print(native_out)
    print(poptorch_out)

    helpers.assert_allclose(actual=poptorch_out, expected=native_out)
