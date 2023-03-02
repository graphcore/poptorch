#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# Tests for PyG torch_scatter ops integration with PopTorch
import json
from torch import gather
import torch
import pytest
import helpers
import poptorch

if helpers.is_running_tests:
    from torch_scatter import scatter, scatter_log_softmax, scatter_softmax, scatter_std, scatter_add, scatter_max
else:

    def scatter():
        pass

    def scatter_log_softmax():
        pass

    def scatter_softmax():
        pass

    def scatter_std():
        pass

    def scatter_add():
        pass

    def scatter_max():
        pass


expected_ops_after_fuse = {
    'scatter': 2,
    'scatter_add': 2,
    'scatter_max': 2,
    'scatter_softmax': 3,
    'scatter_log_softmax': 3,
    'scatter_std': 3,
    'gather': 2
}

expected_group_size_after_fuse = {
    'scatter': 3,
    'scatter_add': 3,
    'scatter_max': 3,
    'scatter_softmax': 3,
    'scatter_log_softmax': 3,
    'scatter_std': 6,
    'gather': 3
}


def check_is_fused(poptorch_model, op_type, expected_group_size,
                   expected_num_ops):
    all_ops = json.loads(poptorch_model._debugGetPopartIR())['maingraph']  # pylint: disable=protected-access
    op_type = "Gather" if op_type == "gather" else "ScatterReduce"
    ops = [op for op in all_ops if op['type'] == op_type]

    assert len(ops) == expected_num_ops
    assert int(ops[0]['attributes']['group_size']) == expected_group_size


def torch_fusible_model(func, src, index, dtype):

    # We do the shape inference from scatter here because we don't support
    # dynamic shaped tensors on the ipu

    dim = 0
    dim_size = int(index.max()) + 1

    class Model(torch.nn.Module):
        def forward(self, src, index, dtype):
            ones = torch.ones_like(src, dtype=dtype)
            two = torch.ones_like(src) * 2
            if func == gather:
                out = func(src, dim, index)
                out_ones = func(ones, dim, index)
                out_two = func(two, dim, index)
            else:
                out = func(src, index, dim_size=dim_size)
                out_ones = func(ones, index, dim_size=dim_size)
                out_two = func(two, index, dim_size=dim_size)
            if isinstance(out, tuple):
                out = out[0]
                out_ones = out_ones[0]
                out_two = out_two[0]

            src_updated = src - torch.sum(out)
            # Functions which should not be fused
            out_updated_s, _ = scatter_max(src_updated,
                                           index,
                                           dim_size=dim_size)
            out_updated_g = gather(src_updated, dim, index)
            out_updated_sum = torch.sum(out_updated_g) + torch.sum(
                out_updated_s)
            return (out_ones + out_two) / out_updated_sum

    model = Model()
    options = poptorch.Options()
    poptorch_model = poptorch.inferenceModel(model, options=options)

    ones = torch.ones_like(src, dtype=dtype)
    two = torch.ones_like(src) * 2

    if func == gather:
        native_out = func(src, dim, index)
        native_out_ones = func(ones, dim, index)
        native_out_two = func(two, dim, index)
    else:
        native_out = func(src, index, dim_size=dim_size)
        native_out_ones = func(ones, index, dim_size=dim_size)
        native_out_two = func(two, index, dim_size=dim_size)
    if isinstance(native_out, tuple):
        native_out = native_out[0]
        native_out_ones = native_out_ones[0]
        native_out_two = native_out_two[0]

    src_updated = src - torch.sum(native_out)
    native_out_updated_s, _ = scatter_max(src_updated, index)
    native_out_updated_g = gather(src_updated, dim, index)
    native_out_updated_sum = torch.sum(native_out_updated_s) + torch.sum(
        native_out_updated_g)

    expected_nat = (native_out_ones + native_out_two) / native_out_updated_sum

    ipu_out = poptorch_model(src, index, dtype)
    # Verify that the ops have been fused
    expected_num_ops = expected_ops_after_fuse[func.__name__]
    expected_group_size = expected_group_size_after_fuse[func.__name__]
    if dtype != torch.float32:
        expected_group_size = expected_group_size - 1
        expected_num_ops = expected_num_ops + 1
    check_is_fused(poptorch_model, func.__name__, expected_group_size,
                   expected_num_ops)

    helpers.assert_allclose(actual=torch.nan_to_num(ipu_out),
                            expected=torch.nan_to_num(expected_nat))


@pytest.mark.parametrize("shape", [(3, ), (3, 5), (3, 5, 5)])
@pytest.mark.parametrize("func", [
    scatter, scatter_add, scatter_max, scatter_softmax, scatter_log_softmax,
    scatter_std, gather
])
@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16])  #, torch.int])
@helpers.overridePoptorchLogLevel('TRACE')
def test_fuse(shape, func, dtype):
    if dtype != torch.float32 and func in [
            scatter_softmax, scatter_log_softmax, scatter_std
    ]:
        pytest.skip("can only be computed with fp32 data types")

    torch.manual_seed(0)
    x = torch.rand(shape)

    ind = torch.randint(3, shape)

    torch_fusible_model(func, x, ind, dtype)
