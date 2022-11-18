#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import helpers
import poptorch


def test_sharded_execution():
    class Model(torch.nn.Module):
        def forward(self, x):
            with poptorch.Block("0", ipu_id=0):
                x = x * 2
            with poptorch.Block("1", ipu_id=1):
                x = x * 3
            with poptorch.Block("2", ipu_id=2):
                x = x * 4
            with poptorch.Block("3", ipu_id=3):
                x = x * 5
            return x

    native = Model()
    stages = [poptorch.Stage(f"{k}") for k in range(0, 4)]
    strategy = poptorch.ShardedExecution(*stages)

    opts = poptorch.Options()
    opts.setExecutionStrategy(strategy)
    ipu = poptorch.inferenceModel(native, opts)

    torch.manual_seed(42)
    inp = torch.randn(3, 7)

    native_out = native(inp)
    ipu_out = ipu(inp)
    helpers.assert_allclose(actual=ipu_out, expected=native_out)
