#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
import helpers
import poptorch


def test_requires_grad_false_simple():
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self, a, b, c, d):
            super().__init__()
            self.a = torch.nn.Parameter(a)
            self.b = torch.nn.Parameter(b)
            self.c = torch.nn.Parameter(c, requires_grad=False)
            self.d = torch.nn.Parameter(d, requires_grad=False)
            self.loss = torch.nn.MSELoss()

        def forward(self, target):
            s0 = self.a + self.b
            s1 = self.c + self.d
            return self.loss(s0 + s1, target)

    # Ends up with requires_grad=True.
    a = torch.randn(5)
    b = torch.randn(5)
    # Ends up with requires_grad=False.
    c = torch.randn(5)
    d = torch.randn(5)
    target = torch.randn(5)

    model = Model(a.clone(), b.clone(), c.clone(), d.clone())
    native_out = model(target)

    poptorch_model = poptorch.trainingModel(model)
    poptorch_out = poptorch_model(target)
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    for _ in range(100):
        poptorch_out = poptorch_model(target)
        assert not torch.allclose(poptorch_out, native_out)
        # 'a' and 'b' are updated
        assert not torch.allclose(poptorch_model.a.data, a)
        assert not torch.allclose(poptorch_model.b.data, b)
        # 'c' and 'd' are not updated
        helpers.assert_allclose(actual=poptorch_model.c.data, expected=c)
        helpers.assert_allclose(actual=poptorch_model.d.data, expected=d)


def test_requires_grad_false_on_single_input():
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.a = torch.nn.Parameter(a)
            self.b = torch.nn.Parameter(b, requires_grad=False)
            self.loss = torch.nn.MSELoss()

        def forward(self, target):
            s = self.a + self.b
            return self.loss(s, target)

    # Ends up with requires_grad=True.
    a = torch.randn(5)
    # Ends up with requires_grad=False.
    b = torch.randn(5)
    target = torch.randn(5)

    model = Model(a.clone(), b.clone())
    native_out = model(target)

    poptorch_model = poptorch.trainingModel(model)
    poptorch_out = poptorch_model(target)
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    for _ in range(100):
        poptorch_out = poptorch_model(target)
        assert not torch.allclose(poptorch_out, native_out)
        # 'a' is updated
        assert not torch.allclose(poptorch_model.a.data, a)
        # 'b' is not updated
        helpers.assert_allclose(actual=poptorch_model.b.data, expected=b)
