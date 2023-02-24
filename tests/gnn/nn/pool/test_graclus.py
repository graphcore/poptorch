# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from torch_geometric.nn import graclus
from torch_geometric.testing import withPackage

from pool_utils import pool_harness


@withPackage('torch_cluster')
def test_graclus(request):
    # E       NotImplementedError: The following operation failed in the TorchScript interpreter.
    # E       Traceback of TorchScript (most recent call last):
    # E         File "/localdata/krzysztofk/workspace/poptorch_view/build/buildenv/lib/python3.8/site-packages/torch_cluster/graclus.py", line 33, in graclus_cluster
    # E
    # E           if num_nodes is None:
    # E               num_nodes = max(int(row.max()), int(col.max())) + 1
    # E                               ~~~ <--- HERE
    # E
    # E           # Remove self-loops.
    # E       RuntimeError: Could not run 'aten::_local_scalar_dense' with arguments from the 'Meta' backend
    pytest.skip(
        f"{request.node.nodeid}: Error: 'torch_cluster/graclus.py, "
        "line 33, in graclus_cluster, RuntimeError: Could not run "
        "'aten::_local_scalar_dense' with arguments from the 'Meta' backend'. "
        "Will be enabled after AFS-144 is fixed.")

    edge_index = torch.tensor([[0, 1], [1, 0]])
    out = pool_harness(graclus, [edge_index])
    assert out.tolist() == [0, 0]
