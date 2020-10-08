#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import poptorch
import pytest


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_old_api():
    class Model(torch.nn.Module):
        def forward(self, x):
            with poptorch.IPU(0):
                x = x + 4
            with poptorch.IPU(1):
                x = x + 2
            return x

    m = Model()

    opts = poptorch.Options()
    opts = poptorch.Options().deviceIterations(2)

    m = poptorch.inferenceModel(Model(), opts)
    m(torch.randn(2, 5))


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_new_api():
    class Model(torch.nn.Module):
        def forward(self, x):
            with poptorch.Phase(0, 0):
                x = x + 4
            with poptorch.Phase(1, 1):
                x = x + 2
            return x

    m = Model()

    opts = poptorch.Options()
    opts = poptorch.Options().deviceIterations(2)

    m = poptorch.inferenceModel(Model(), opts)
    m(torch.randn(2, 5))


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_new_api_wrap():
    class Block(torch.nn.Module):
        def forward(self, x):
            return x * 6

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = Block()
            self.l2 = Block()

        def forward(self, x):
            x = self.l1(x)
            x = self.l2(x)
            return x

    m = Model()
    m.l1 = poptorch.BeginPhase(ipu_id=0,
                               phase_id=0,
                               layer_to_call=m.l1,
                               stage_id=0)
    m.l2 = poptorch.BeginPhase(ipu_id=1,
                               phase_id=1,
                               layer_to_call=m.l1,
                               stage_id=1)

    opts = poptorch.Options()
    opts = poptorch.Options().deviceIterations(2)

    m = poptorch.inferenceModel(Model(), opts)
    m(torch.randn(2, 5))


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_old_api_wrap():
    class Block(torch.nn.Module):
        def forward(self, x):
            return x * 6

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = Block()
            self.l2 = Block()

        def forward(self, x):
            x = self.l1(x)
            x = self.l2(x)
            return x

    m = Model()
    m.l1 = poptorch.IPU(0, m.l1)
    m.l2 = poptorch.IPU(1, m.l2)

    opts = poptorch.Options()
    opts = poptorch.Options().deviceIterations(2)

    m = poptorch.inferenceModel(Model(), opts)
