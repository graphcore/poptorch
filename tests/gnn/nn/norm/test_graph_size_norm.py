# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch

from norm_utils import norm_harness

from torch_geometric.nn import GraphSizeNorm


def test_graph_size_norm(request):

    pytest.skip(
        f"{request.node.nodeid}: Error: 'Could not run "
        "'aten::_local_scalar_dense' with arguments from the 'Meta' backend'."
        " Will be enabled after AFS-144 is fixed.")

    x = torch.randn(100, 16)
    batch = torch.repeat_interleave(torch.full((10, ), 10, dtype=torch.long))

    norm = GraphSizeNorm()
    assert str(norm) == 'GraphSizeNorm()'

    out = norm_harness(norm, [x, batch])
    assert out.size() == (100, 16)
