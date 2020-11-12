#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch
import pytest

torch.manual_seed(42)
params_einsum = [
    ('i,j->j', [torch.randn(5), torch.randn(4)]),
    ('i,j->ji', [torch.randn(5), torch.randn(4)]),
    ('bij,bjk->bik', [torch.randn(3, 2, 5),
                      torch.randn(3, 5, 4)]),
    ('bn,anm,bm->ba',
     [torch.randn(2, 5),
      torch.randn(3, 5, 4),
      torch.randn(2, 4)]),
    ('bfnd,ndh->bfh', [torch.randn(2, 3, 4, 5),
                       torch.randn(4, 5, 6)]),
    ('nmku,buvm->bnkv', [torch.randn(2, 3, 4, 5),
                         torch.randn(6, 5, 7, 3)])
]


@pytest.mark.parametrize("params", params_einsum)
@pytest.mark.parametrize("implicit_rhs", {True, False})
def test_einsum(params, implicit_rhs):
    class Model(torch.nn.Module):
        def forward(self):
            eq = params[0].split('->')[0] if implicit_rhs else params[0]
            return torch.einsum(eq, params[1])

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    # Run on CPU
    native_out = model()

    # Run on IPU
    poptorch_out = poptorch_model()

    assert native_out.size() == poptorch_out.size()
    torch.testing.assert_allclose(native_out, poptorch_out)
