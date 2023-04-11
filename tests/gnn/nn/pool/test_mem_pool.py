# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from torch_geometric.nn import MemPooling
from torch_geometric.utils import to_dense_batch

from pool_utils import pool_harness


@pytest.mark.skip(reason="TODO(AFS-256)")
def test_mem_pool():
    mpool1 = MemPooling(4, 8, heads=3, num_clusters=2)
    assert mpool1.__repr__() == 'MemPooling(4, 8, heads=3, num_clusters=2)'

    x = torch.randn(17, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    _, mask = to_dense_batch(x, batch)

    out1, S = pool_harness(mpool1, [x, batch])
    loss = MemPooling.kl_loss(S)
    with torch.autograd.set_detect_anomaly(True):
        loss.backward()

    assert out1.size() == (5, 2, 8)
    assert S[~mask].sum() == 0
    assert round(S[mask].sum().item()) == x.size(0)
    assert float(loss) > 0


@pytest.mark.skip(reason="TODO(AFS-256)")
def test_mem_pool_chain():
    mpool1 = MemPooling(4, 8, heads=3, num_clusters=2)
    assert mpool1.__repr__() == 'MemPooling(4, 8, heads=3, num_clusters=2)'
    mpool2 = MemPooling(8, 4, heads=2, num_clusters=1)
    assert mpool2.__repr__() == 'MemPooling(8, 4, heads=2, num_clusters=1)'

    x = torch.randn(17, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4])

    out1, _ = mpool1(x, batch)
    out2, _ = pool_harness(mpool2, [out1])

    assert out2.size() == (5, 1, 4)
