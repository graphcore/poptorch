#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import poptorch
import pytest


def test_api_inline():
    class Model(torch.nn.Module):
        def forward(self, x):
            poptorch.Block.useAutoId()
            with poptorch.Block(ipu_id=0):
                x = x + 4
            with poptorch.Block(ipu_id=1):
                x = x + 2
            return x

    m = Model()

    opts = poptorch.Options()
    opts = poptorch.Options().deviceIterations(2)

    m = poptorch.inferenceModel(Model(), opts)
    m(torch.randn(2, 5))


def test_api_wrap():
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
    m.l1 = poptorch.BeginBlock(m.l1)
    m.l2 = poptorch.BeginBlock(m.l2)

    opts = poptorch.Options()
    opts = poptorch.Options().deviceIterations(2)

    m = poptorch.inferenceModel(Model(), opts)
    m(torch.randn(2, 5))
