# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from torch_geometric.nn import MemPooling
from torch_geometric.utils import to_dense_batch

from pool_utils import op_harness


def test_mem_pool(request):
    pytest.skip(
        f"{request.node.nodeid}: Error: 'NotImplementedError: Could "
        "not run 'aten::_local_scalar_dense' with arguments from the 'Meta' "
        "backend'. Will be enabled after AFS-144 is fixed.")

    mpool1 = MemPooling(4, 8, heads=3, num_clusters=2)
    assert mpool1.__repr__() == 'MemPooling(4, 8, heads=3, num_clusters=2)'

    x = torch.randn(17, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    _, mask = to_dense_batch(x, batch)

    out1, S = op_harness(mpool1, [x, batch])
    loss = MemPooling.kl_loss(S)
    with torch.autograd.set_detect_anomaly(True):
        loss.backward()

    assert out1.size() == (5, 2, 8)
    assert S[~mask].sum() == 0
    assert round(S[mask].sum().item()) == x.size(0)
    assert float(loss) > 0


def test_mem_pool_chain(request):
    pytest.skip(
        f"{request.node.nodeid}: Error: "
        "'torch/_meta_registrations.py:708 TypeError: '>=' not supported "
        "between instances of 'NoneType' and 'int''. Will be enabled after "
        "AFS-141 is fixed.")

    mpool1 = MemPooling(4, 8, heads=3, num_clusters=2)
    assert mpool1.__repr__() == 'MemPooling(4, 8, heads=3, num_clusters=2)'
    mpool2 = MemPooling(8, 4, heads=2, num_clusters=1)
    assert mpool2.__repr__() == 'MemPooling(8, 4, heads=2, num_clusters=1)'

    x = torch.randn(17, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4])

    out1, _ = mpool1(x, batch)
    out2, _ = op_harness(mpool2, [out1])

    assert out2.size() == (5, 1, 4)
