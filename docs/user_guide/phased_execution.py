#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import torch.nn.functional as F
import poptorch

# pylint: disable=function-redefined, too-many-function-args
# annotations_start
poptorch.setLogLevel(1)  # Force debug logging
N = 3
size = 10


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = []
        for n in range(N * 6):
            weight = torch.nn.Parameter(torch.rand(size, size),
                                        requires_grad=True)
            self.register_parameter(f"w{n}", weight)
            self.weights.append(weight)

    def forward(self, in0, target=None):
        phase = 0
        weight = iter(self.weights)
        with poptorch.Block("phase0_ipu0"):
            ins = torch.split(in0, size)
        for n in range(N * 3):
            out = []
            for ipu in range(2):
                x = ins[ipu]
                with poptorch.Block(f"phase{phase}_ipu{ipu}"):
                    x = torch.matmul(next(weight), x)
                    out.append(F.relu(x))
            ins = out[1], out[0]
            # We want 2 matmuls in the same phase
            if n % 3 != 1:
                phase += 1
        with poptorch.Block(f"phase{N*2-1}_ipu1"):
            res = ins[0] + ins[1]
            if target is None:
                return res
            return res, torch.nn.L1Loss(reduction="mean")(res, target)


input = torch.rand(size * 2, 1)
target = torch.rand(size, 1)
model = Model()
opts = poptorch.Options()
phases = []
# Alternate between 0-2 and 1-3
for n in range(N):
    phases.append([
        poptorch.Stage(f"phase{2*n}_ipu0").ipu(0),
        poptorch.Stage(f"phase{2*n}_ipu1").ipu(2)
    ])
    phases.append([
        poptorch.Stage(f"phase{2*n+1}_ipu0").ipu(1),
        poptorch.Stage(f"phase{2*n+1}_ipu1").ipu(3)
    ])
opts.setExecutionStrategy(poptorch.ParallelPhasedExecution(*phases))
poptorch_model = poptorch.trainingModel(model, opts)
poptorch_model.compile(input, target)

# annotations_end


# stage_start
class Model(torch.nn.Module):
    def forward(self, x, y):
        with poptorch.Block("A"):
            c = x + x
        with poptorch.Block("B"):
            d = y + y
        with poptorch.Block("C"):
            e = x * 3

        return c, d, e


first = poptorch.Phase(poptorch.Stage("A").ipu(0))
# Regrouped in a single stage
second = poptorch.Phase(poptorch.Stage("B", "C").ipu(1))
# 2 separate stages
second = poptorch.Phase(poptorch.Stage("B").ipu(1), poptorch.Stage("C").ipu(3))
# stage_end

opts = poptorch.Options()
opts.autoRoundNumIPUs(True)
opts.setExecutionStrategy(poptorch.ParallelPhasedExecution(first, second))
m = poptorch.inferenceModel(Model(), opts)
m.compile(input, input)
m.destroy()


class Model(torch.nn.Module):
    def forward(self, x, y):
        with poptorch.Block("A"):
            c = x + x
        with poptorch.Block("A2"):
            d = y + y

        with poptorch.Block("B"):
            e = c + d
        with poptorch.Block("B2"):
            f = y + d

        with poptorch.Block("C"):
            g = e + f
        with poptorch.Block("C2"):
            h = f + y
        return g, h


opts = poptorch.Options()
# serial_start
strategy = poptorch.SerialPhasedExecution(
    poptorch.Phase(poptorch.Stage("A"), poptorch.Stage("A2")),
    poptorch.Phase(poptorch.Stage("B"), poptorch.Stage("B2")),
    poptorch.Phase(poptorch.Stage("C"), poptorch.Stage("C2")))

strategy.phase(0).ipus(0, 1)
strategy.phase(1).ipus(0, 1)
strategy.phase(2).ipus(0, 1)

opts.setExecutionStrategy(strategy)
# serial_end

m = poptorch.inferenceModel(Model(), opts)
m.compile(input, input)
m.destroy()


class Model(torch.nn.Module):
    def forward(self, x, y):
        poptorch.Block.useAutoId()
        with poptorch.Block():
            c = x + x
        with poptorch.Block():
            d = y + y

        with poptorch.Block():
            e = c + d
        with poptorch.Block():
            f = y + d

        with poptorch.Block():
            g = e + f
        with poptorch.Block():
            h = f + y
        return g, h


opts = poptorch.Options()
# parallel_start
strategy = poptorch.ParallelPhasedExecution(
    poptorch.Phase(poptorch.Stage("0"), poptorch.Stage("1")),
    poptorch.Phase(poptorch.Stage("2"), poptorch.Stage("3")),
    poptorch.Phase(poptorch.Stage("4"), poptorch.Stage("5")))

strategy.phase(0).ipus(0, 2)
strategy.phase(1).ipus(1, 3)
strategy.phase(2).ipus(0, 2)

opts.setExecutionStrategy(strategy)
# parallel_end
m = poptorch.inferenceModel(Model(), opts)
m.compile(input, input)
m.destroy()
