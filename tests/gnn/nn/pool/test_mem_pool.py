# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch

from torch_geometric.nn import MemPooling
from torch_geometric.utils import to_dense_batch

from pool_utils import pool_harness

import helpers
import poptorch


def test_mem_pool_basic():
    torch.manual_seed(42)

    mpool1 = MemPooling(4, 8, heads=3, num_clusters=2)
    assert mpool1.__repr__() == 'MemPooling(4, 8, heads=3, num_clusters=2)'

    x = torch.randn(17, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    _, mask = to_dense_batch(x, batch)

    batch_size = int(batch.max() + 1)
    out1, S = pool_harness(mpool1, [x, batch, None, 4, batch_size])
    assert out1.size() == (5, 2, 8)
    assert S[~mask].sum() == 0
    assert round(S[mask].sum().item()) == x.size(0)


def test_mem_pool_basic_custom_loss():
    torch.manual_seed(42)

    x = torch.randn(17, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4])

    class MemPoolWrapper(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.op = MemPooling(4, 8, heads=3, num_clusters=2)

        def forward(self, *args, **kwargs):
            out1, S = self.op.forward(*args, **kwargs)
            loss = MemPooling.kl_loss(S)
            return out1, poptorch.identity_loss(loss, "sum")

    model = MemPoolWrapper()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    poptorch_model = poptorch.trainingModel(model, optimizer=optimizer)
    batch_size = int(batch.max() + 1)
    out1_expected, _ = model(x, batch, None, 4, batch_size)
    out1, loss = poptorch_model(x, batch, None, 4, batch_size)

    assert float(loss) > 0
    assert out1.size() == (5, 2, 8)
    helpers.assert_allclose(actual=out1, expected=out1_expected)


def test_mem_pool_chain():
    torch.manual_seed(42)

    mpool1 = MemPooling(4, 8, heads=3, num_clusters=2)
    assert mpool1.__repr__() == 'MemPooling(4, 8, heads=3, num_clusters=2)'
    mpool2 = MemPooling(8, 4, heads=2, num_clusters=1)
    assert mpool2.__repr__() == 'MemPooling(8, 4, heads=2, num_clusters=1)'

    x = torch.randn(17, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4])

    out1, _ = mpool1(x, batch)
    assert out1.size() == (5, 2, 8)
    out2, _ = pool_harness(mpool2, [out1])
    assert out2.size() == (5, 1, 4)
