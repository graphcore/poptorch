# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from pool_utils import pool_harness


def test_global_pool():
    N_1, N_2 = 4, 6
    x = torch.randn(N_1 + N_2, 4)
    batch = torch.tensor([0 for _ in range(N_1)] + [1 for _ in range(N_2)])

    out = pool_harness(global_add_pool, [x, batch, 2])
    assert out.size() == (2, 4)
    torch.testing.assert_close(out[0], x[:4].sum(dim=0))
    torch.testing.assert_close(out[1], x[4:].sum(dim=0))

    out = pool_harness(global_add_pool, [x, None])
    assert out.size() == (1, 4)
    torch.testing.assert_close(out, x.sum(dim=0, keepdim=True))

    out = pool_harness(global_mean_pool, [x, batch, 2])
    assert out.size() == (2, 4)
    torch.testing.assert_close(out[0], x[:4].mean(dim=0))
    torch.testing.assert_close(out[1], x[4:].mean(dim=0))

    out = pool_harness(global_mean_pool, [x, None])
    assert out.size() == (1, 4)
    torch.testing.assert_close(out, x.mean(dim=0, keepdim=True))

    out = pool_harness(global_max_pool, [x, batch, 2])
    assert out.size() == (2, 4)
    torch.testing.assert_close(out[0], x[:4].max(dim=0)[0])
    torch.testing.assert_close(out[1], x[4:].max(dim=0)[0])


@pytest.mark.skip(reason="TODO(AFS-140)")
def test_global_max_pool_no_batch():
    N_1, N_2 = 4, 6
    x = torch.randn(N_1 + N_2, 4)

    out = pool_harness(global_max_pool, [x, None])
    assert out.size() == (1, 4)
    torch.testing.assert_close(out, x.max(dim=0, keepdim=True)[0])


def test_permuted_global_pool():
    N_1, N_2 = 4, 6
    x = torch.randn(N_1 + N_2, 4)
    batch = torch.cat([torch.zeros(N_1), torch.ones(N_2)]).to(torch.long)
    perm = torch.randperm(N_1 + N_2)

    px = x[perm]
    pbatch = batch[perm]
    px1 = px[pbatch == 0]
    px2 = px[pbatch == 1]

    out = pool_harness(global_add_pool, [px, pbatch, 2])
    assert out.size() == (2, 4)
    assert torch.allclose(out[0], px1.sum(dim=0))
    assert torch.allclose(out[1], px2.sum(dim=0))

    out = pool_harness(global_mean_pool, [px, pbatch, 2])
    assert out.size() == (2, 4)
    assert torch.allclose(out[0], px1.mean(dim=0))
    assert torch.allclose(out[1], px2.mean(dim=0))

    out = pool_harness(global_max_pool, [px, pbatch, 2])
    assert out.size() == (2, 4)
    assert torch.allclose(out[0], px1.max(dim=0)[0])
    assert torch.allclose(out[1], px2.max(dim=0)[0])


def test_dense_global_pool():
    x = torch.randn(3, 16, 32)
    out = pool_harness(global_add_pool, [x, None])
    assert torch.allclose(out, x.sum(dim=1))
