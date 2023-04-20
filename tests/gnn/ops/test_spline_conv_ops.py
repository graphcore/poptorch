#!/usr/bin/env python3
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

# Tests for PyG torch_spline_conv ops integration with PopTorch
from collections import namedtuple
from copy import deepcopy
import torch
import pytest
import helpers
import poptorch

if helpers.is_running_tests:
    from torch_spline_conv import spline_basis, spline_weighting
else:

    def spline_basis():
        pass

    def spline_weighting():
        pass


def gen_basis_input_data(num_edges, num_dims, max_kernel_size, dtype):
    torch.manual_seed(0)
    pseudo = torch.rand(num_edges, num_dims, dtype=dtype)
    kernel_size = torch.randint(1, max_kernel_size, (num_dims, ))
    is_open_spline = torch.randint(0, 2, (num_dims, ), dtype=torch.uint8)
    return pseudo, kernel_size, is_open_spline


BasisParams = namedtuple('BasisParams', 'edges dims max_kernel_size degree')
test_params_b = (BasisParams(6, 2, 6, 1), BasisParams(64, 3, 16, 3))


@pytest.mark.parametrize("params", test_params_b)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_spline_basis(params, dtype):
    class Model(torch.nn.Module):
        def __init__(self, degree):
            self.degree = degree
            super().__init__()

        def forward(self, p, ks, ios):
            return spline_basis(p, ks, ios, self.degree)

    *params, degree = params
    pseudo, kernel_size, is_open_spline = gen_basis_input_data(*params, dtype)

    model = Model(degree)
    pseudo_f32 = pseudo.type(torch.float32)
    basis, weight_index = model(pseudo_f32, kernel_size, is_open_spline)
    reference_output = (basis.type(dtype), weight_index)

    poptorch_model = poptorch.inferenceModel(deepcopy(model))
    poptorch_output = poptorch_model(pseudo, kernel_size, is_open_spline)

    atol, rtol = (1e-3, 1e-5) if dtype == torch.float16 else (1e-5, 1e-8)
    helpers.assert_allclose(actual=poptorch_output,
                            expected=reference_output,
                            atol=atol,
                            rtol=rtol)


def gen_weighting_input_data(edges, in_ch, out_ch, kernel_size, num_splines,
                             dtype):
    torch.manual_seed(0)
    x = torch.rand(edges, in_ch, dtype=dtype)
    weights = torch.rand(kernel_size, in_ch, out_ch, dtype=dtype)
    basis = torch.rand(edges, num_splines, dtype=dtype)
    weight_index = torch.randint(0, kernel_size, (edges, num_splines))
    return x, weights, basis, weight_index


WeightingParams = namedtuple('WeightingParams',
                             'edges in_ch out_ch kernel_size num_splines')
test_params_w = (WeightingParams(6, 4, 4, 10,
                                 8), WeightingParams(24, 5, 6, 3, 10))


@pytest.mark.parametrize("params", test_params_w)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_spline_weighting(params, dtype):
    class Model(torch.nn.Module):
        def forward(self, x, weight, basis, weight_index):
            return spline_weighting(x, weight, basis, weight_index)

    x, weight, basis, weight_index = gen_weighting_input_data(*params, dtype)

    model = Model()
    x_f32 = x.type(torch.float32)
    weight_f32 = weight.type(torch.float32)
    basis_f32 = basis.type(torch.float32)
    reference_output = model(x_f32, weight_f32, basis_f32, weight_index)

    poptorch_model = poptorch.inferenceModel(deepcopy(model))
    weight_index = weight_index.type(torch.int32)
    poptorch_output = poptorch_model(x, weight, basis, weight_index)

    atol, rtol = (1e-2, 1e-3) if dtype == torch.float16 else (1e-5, 1e-8)
    helpers.assert_allclose(actual=poptorch_output,
                            expected=reference_output.type(dtype),
                            atol=atol,
                            rtol=rtol)
