# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
from torch_geometric.nn.functional import bro
import poptorch


@pytest.mark.skip(reason="TODO(AFS-269)")
def test_bro():
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2])

    g1 = torch.tensor([
        [0.2, 0.2, 0.2, 0.2],
        [0.0, 0.2, 0.2, 0.2],
        [0.2, 0.0, 0.2, 0.2],
        [0.2, 0.2, 0.0, 0.2],
    ])

    g2 = torch.tensor([
        [0.2, 0.2, 0.2, 0.2],
        [0.0, 0.2, 0.2, 0.2],
        [0.2, 0.0, 0.2, 0.2],
    ])

    g3 = torch.tensor([
        [0.2, 0.2, 0.2, 0.2],
        [0.2, 0.0, 0.2, 0.2],
    ])

    class Model(torch.nn.Module):
        def forward(self, g1, g2, g3, batch):
            return bro(torch.cat([g1, g2, g3], dim=0), batch)

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    ipu_out = poptorch_model(g1, g2, g3, batch)

    s = 0.
    for g in [torch.cat([g1, g2, g3]) / 3]:
        s += torch.norm(g @ g.t() - torch.eye(g.shape[0]), p=2)

    assert torch.isclose(s / 3., ipu_out)
