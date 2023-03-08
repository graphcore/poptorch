# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch
from torch_geometric.nn.functional import gini

import poptorch


def test_gini():

    w = torch.tensor([[0., 0., 0., 0.], [0., 0., 0., 1000.0]])

    class Model(torch.nn.Module):
        def forward(self, w):
            return gini(w)

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    ipu_out = poptorch_model(w)

    assert torch.isclose(ipu_out, torch.tensor(0.5))
