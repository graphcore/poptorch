#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
import torch
import torch.nn as nn
import poptorch
import pytest

MANY_TYPES = (torch.float32, torch.float64, torch.int32, torch.int64)

DEMOTED_ON_IPU = (torch.float64, torch.int64)


def get_simple_adder(return_type):
    class SimpleAdder(nn.Module):
        def forward(self, x, y):
            return (x + y).type(return_type)

    return poptorch.inferenceModel(SimpleAdder())


@pytest.mark.parametrize("input_type, output_type",
                         zip(MANY_TYPES, MANY_TYPES))
def test_many_input_output_types(input_type, output_type):
    model = get_simple_adder(output_type)
    t1 = torch.tensor([1.0, 25, -1.0, 83], dtype=input_type)
    t2 = torch.tensor([2.0, 35, 1.0, 32.4], dtype=input_type)

    output = model(t1, t2)

    if output_type not in DEMOTED_ON_IPU:
        assert output[0].dtype == output_type
        assert output[1].dtype == output_type

    assert np.all(
        np.isclose(output.numpy(), np.array([3., 60., 0., 115.4]), atol=0.5))


def get_simple_add_two():
    class GetSimpleAddTwo(nn.Module):
        def forward(self, x):
            return x + 2

    return poptorch.inferenceModel(GetSimpleAddTwo())


@pytest.mark.parametrize("input_type", MANY_TYPES)
@pytest.mark.skip(reason="TODO(T21014)")
def test_add_two_many_types(input_type):
    model = get_simple_add_two()

    t = torch.tensor([1.0, 25., -1.0, 83.], dtype=input_type)
    assert np.all(
        np.isclose(model(t).numpy(), np.array([3.0, 27., 1, 85.]), atol=0.5))
