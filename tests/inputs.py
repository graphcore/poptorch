#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import poptorch


def test_simple_tuple():
    class SimpleAdder(nn.Module):
        def forward(self, t):
            assert isinstance(t, tuple)
            (x, y) = t
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            return x + y

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])

    assert inference_model((t1, t2)) == 3.0


def test_nested_tuples():
    class SimpleAdder(nn.Module):
        def forward(self, tpl1, t2, tpl3456):
            (t1, ) = tpl1
            (t3, (t4, t5), t6) = tpl3456
            assert isinstance(t1, torch.Tensor)
            assert isinstance(t2, torch.Tensor)
            assert isinstance(t3, torch.Tensor)
            assert isinstance(t4, torch.Tensor)
            assert isinstance(t5, torch.Tensor)
            assert isinstance(t6, torch.Tensor)
            return t1 + t2 + t3 + t4 + t5 + t6

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])
    t3 = torch.tensor([3.])
    t4 = torch.tensor([4.])
    t5 = torch.tensor([5.])
    t6 = torch.tensor([6.])

    assert inference_model((t1, ), t2, (t3, (t4, t5), t6)) == 21.0


def test_optional_inputs():
    class SimpleAdder(nn.Module):
        def forward(self, t1, t2, t3=torch.ones(1), t4=torch.zeros(1)):
            return t1 * t3 + t2 * t4

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])
    t4 = torch.tensor([4.])

    assert inference_model(t1, t2) == 1.0
    assert inference_model(t1, t2, t4=t4) == 9.0
    assert inference_model(t4=t4, t1=t1, t2=t2) == 9.0


def test_list_inputs():
    class SimpleAdder(nn.Module):
        def forward(self, t1, t2, x):
            l = [t1, t2]
            x = l[0] + x
            l[1] = x
            return l

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])
    t4 = torch.tensor([4.])

    expected = [torch.tensor([1.0]), torch.tensor([5.0])]

    assert inference_model(t1, t2, t4) == expected
