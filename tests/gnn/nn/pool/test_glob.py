# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from pool_utils import pool_harness


def test_global_pool(request):
    pytest.skip(
        f"{request.node.nodeid}: Error: 'NotImplementedError: Could not run "
        "'aten::_local_scalar_dense' with arguments from the 'Meta' backend'. "
        "Will be enabled after AFS-144 is fixed.")

    N_1, N_2 = 4, 6
    x = torch.randn(N_1 + N_2, 4)
    batch = torch.tensor([0 for _ in range(N_1)] + [1 for _ in range(N_2)])

    out = pool_harness(global_add_pool, [x, batch])
    assert out.size() == (2, 4)
    assert out[0].tolist() == x[:4].sum(dim=0).tolist()
    assert out[1].tolist() == x[4:].sum(dim=0).tolist()

    out = pool_harness(global_add_pool, [x, None])
    assert out.size() == (1, 4)
    assert out.tolist() == x.sum(dim=0, keepdim=True).tolist()

    out = pool_harness(global_mean_pool, [x, batch])
    assert out.size() == (2, 4)
    assert out[0].tolist() == x[:4].mean(dim=0).tolist()
    assert out[1].tolist() == x[4:].mean(dim=0).tolist()

    out = pool_harness(global_mean_pool, [x, None])
    assert out.size() == (1, 4)
    assert out.tolist() == x.mean(dim=0, keepdim=True).tolist()

    out = pool_harness(global_max_pool, [x, batch])
    assert out.size() == (2, 4)
    assert out[0].tolist() == x[:4].max(dim=0)[0].tolist()
    assert out[1].tolist() == x[4:].max(dim=0)[0].tolist()


def test_global_max_pool_no_batch(request):
    pytest.skip(
        f"{request.node.nodeid}: Error: 'poptorch/_poplar_executor.py:"
        "945 In poptorch/source/ImplicitCasting.cpp:141: 'poptorch_cpp_error':"
        " constant->kind() != symbols::poptorch::tensor_constant'. Will be "
        "enabled after AFS-140 is fixed.")

    N_1, N_2 = 4, 6
    x = torch.randn(N_1 + N_2, 4)

    out = pool_harness(global_max_pool, [x, None])
    assert out.size() == (1, 4)
    assert out.tolist() == x.max(dim=0, keepdim=True)[0].tolist()


def test_permuted_global_pool(request):
    pytest.skip(
        f"{request.node.nodeid}: Error: 'NotImplementedError: Could "
        "not run 'aten::_local_scalar_dense' with arguments from the 'Meta' "
        "backend'. Will be enabled after AFS-144 is fixed.")

    N_1, N_2 = 4, 6
    x = torch.randn(N_1 + N_2, 4)
    batch = torch.cat([torch.zeros(N_1), torch.ones(N_2)]).to(torch.long)
    perm = torch.randperm(N_1 + N_2)

    px = x[perm]
    pbatch = batch[perm]
    px1 = px[pbatch == 0]
    px2 = px[pbatch == 1]

    out = pool_harness(global_add_pool, [px, pbatch])
    assert out.size() == (2, 4)
    assert torch.allclose(out[0], px1.sum(dim=0))
    assert torch.allclose(out[1], px2.sum(dim=0))

    out = pool_harness(global_mean_pool, [px, pbatch])
    assert out.size() == (2, 4)
    assert torch.allclose(out[0], px1.mean(dim=0))
    assert torch.allclose(out[1], px2.mean(dim=0))

    out = pool_harness(global_max_pool, [px, pbatch])
    assert out.size() == (2, 4)
    assert torch.allclose(out[0], px1.max(dim=0)[0])
    assert torch.allclose(out[1], px2.max(dim=0)[0])


def test_dense_global_pool():
    x = torch.randn(3, 16, 32)
    out = pool_harness(global_add_pool, [x, None])
    assert torch.allclose(out, x.sum(dim=1))
