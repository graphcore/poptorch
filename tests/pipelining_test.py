#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import poptorch
import helpers


def test_api_inline(capfd):
    poptorch.setLogLevel(1)  # Force debug logging

    class Model(torch.nn.Module):
        def forward(self, x):
            poptorch.Block.useAutoId()
            with poptorch.Block(ipu_id=0):
                x = x * 4
            with poptorch.Block(ipu_id=1):
                x = x * 2
            return x

    m = Model()

    opts = poptorch.Options()
    opts = poptorch.Options().deviceIterations(2)

    m = poptorch.inferenceModel(m, opts)
    m(torch.randn(2, 5))

    log = helpers.LogChecker(capfd)
    log.assert_contains("enablePipelining set to value 1")
    log.assert_contains(" Mul:0 ", " mode(Pipelined), ipu(0), stage(0)")
    log.assert_contains(" Mul:0/1 ", " mode(Pipelined), ipu(1), stage(1)")


def test_api_wrap(capfd):
    """
    stage "0" ipu(0) stage(0) l0 l1 l2
    """
    poptorch.setLogLevel(1)  # Force debug logging

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
    m.l1 = poptorch.BeginBlock(m.l1, ipu_id=0)
    m.l2 = poptorch.BeginBlock(m.l2, ipu_id=0)

    opts = poptorch.Options()
    opts = poptorch.Options().deviceIterations(2)

    m = poptorch.inferenceModel(m, opts)
    m(torch.randn(2, 5))

    log = helpers.LogChecker(capfd)
    log.assert_contains("enablePipelining set to value 0")
    log.assert_contains(" Mul:0 ", " mode(Pipelined), ipu(0), stage(0)")
    log.assert_contains(" Mul:0/1 ", " mode(Pipelined), ipu(0), stage(0)")


def test_api_wrap_2stages(capfd):
    """
    stage "0" ipu(0) stage(0) l0
    stage "1" ipu(1) stage(1) l1 / l2
    """
    poptorch.setLogLevel(1)  # Force debug logging

    class Block(torch.nn.Module):
        def forward(self, x):
            return x * 6

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l0 = Block()
            self.l1 = Block()
            self.l2 = Block()

        def forward(self, x):
            x = self.l0(x)
            x = self.l1(x)
            x = self.l2(x)
            return x

    m = Model()
    m.l1 = poptorch.BeginBlock(m.l1, ipu_id=1)
    m.l2 = poptorch.BeginBlock(m.l2, ipu_id=1)

    opts = poptorch.Options()
    opts = poptorch.Options().deviceIterations(2)

    m = poptorch.inferenceModel(m, opts)
    m(torch.randn(2, 5))

    log = helpers.LogChecker(capfd)
    log.assert_contains("enablePipelining set to value 1")
    log.assert_contains(" Mul:0 ", " mode(Pipelined), ipu(0), stage(0)")
    log.assert_contains(" Mul:0/1 ", " mode(Pipelined), ipu(1), stage(1)")
    log.assert_contains(" Mul:0/2 ", " mode(Pipelined), ipu(1), stage(1)")


def test_inline_AutoIncrement(capfd):
    poptorch.setLogLevel(1)  # Force debug logging

    class Model(torch.nn.Module):
        def forward(self, x):
            poptorch.Block.useAutoId()
            with poptorch.Block(ipu_id=0):
                x = x * 2
            with poptorch.Block(ipu_id=1):
                x = x * 3
            with poptorch.Block(ipu_id=2):
                x = x * 4
            with poptorch.Block(ipu_id=1):
                x = x * 5
            return x

    m = Model()

    opts = poptorch.Options()
    opts = poptorch.Options().deviceIterations(4)
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))

    m = poptorch.inferenceModel(m, opts)
    m.compile(torch.randn(4, 5))

    log = helpers.LogChecker(capfd)
    log.assert_contains("enablePipelining set to value 1")
    log.assert_contains(" Mul:0 ", " mode(Pipelined), ipu(0), stage(1)")
    log.assert_contains(" Mul:0/1 ", " mode(Pipelined), ipu(1), stage(2)")
    log.assert_contains(" Mul:0/2 ", " mode(Pipelined), ipu(2), stage(3)")
    log.assert_contains(" Mul:0/3 ", " mode(Pipelined), ipu(1), stage(4)")


def test_api_AutoIncrement(capfd):
    poptorch.setLogLevel(1)  # Force debug logging

    class Block(torch.nn.Module):
        def forward(self, x):
            return x * 6

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = Block()
            self.l2 = Block()
            self.l3 = Block()
            self.l4 = Block()

        def forward(self, x):
            x = self.l1(x)
            x = self.l2(x)
            x = self.l3(x)
            x = self.l4(x)
            return x

    m = Model()
    m.l2 = poptorch.BeginBlock(m.l2, ipu_id=1)
    m.l3 = poptorch.BeginBlock(m.l3, ipu_id=2)
    m.l4 = poptorch.BeginBlock(m.l4, ipu_id=1)

    opts = poptorch.Options()
    opts = poptorch.Options().deviceIterations(4)
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))

    m = poptorch.inferenceModel(m, opts)
    m(torch.randn(4, 5))

    log = helpers.LogChecker(capfd)
    log.assert_contains("enablePipelining set to value 1")
    log.assert_contains(" Mul:0 ", " mode(Pipelined), ipu(0), stage(0)")
    log.assert_contains(" Mul:0/1 ", " mode(Pipelined), ipu(1), stage(1)")
    log.assert_contains(" Mul:0/2 ", " mode(Pipelined), ipu(2), stage(2)")
    log.assert_contains(" Mul:0/3 ", " mode(Pipelined), ipu(1), stage(3)")
