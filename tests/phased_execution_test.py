#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import torch.nn.functional as F
import pytest
import poptorch
import helpers

# Model: 2x2 S1 ExecutionPhase, repeated N times:
# _____________________________________________________________________________
# phase 0:            IPU 0            |                       IPU 2
# in0 ---- Slice/Slice -----------------------------.
#            |                         |            |
# w0 ----- MatMul                      |          MatMul ----- w1
#            |                         |            |
#          ReLU                        |           ReLU
#            |                         |            |
#            +------------------------.|.-----------+
#______________________________________X__(inter-phase cross-IPU copy)_________
# phase 1:            IPU 1           /|\                      IPU 3
#            .-----------------------' | '----------.
#            |                         |            |
# w2 ----- MatMul                      |          MatMul ----- w3
#            |                         |            |
#          ReLU                        |           ReLU
#            |                         |            |
#            +------------------------.|.-----------+
#                                      X  (intra-phase cross-IPU copy)
#                                     /|\
#            .-----------------------' | '----------.
#            |                         |            |
# w4 ----- MatMul                      |          MatMul ----- w5
#            |                         |            |
#          ReLU                        |           ReLU
#            |                         |            |
#            +------------------------.|.-----------+
#______________________________________X_______________________________________
# phase 2:            IPU 0           /|\                      IPU 2
# ......                               |
# ......                               |
#______________________________________X__(inter-phase cross-IPU copy)_________
# phase N*2-1:        IPU 1           /|\                      IPU 3
#            .-----------------------' | '----------.
#            |                         |            |
# w2 ----- MatMul                      |          MatMul ----- w3
#            |                         |            |
#          ReLU                        |           ReLU
#            |                         |            |
#            +------------------------.|.-----------+
#                                      X  (intra-phase cross-IPU copy)
#                                     /|\
#            .-----------------------' | '----------.
#            |                         |            |
# w4 ----- MatMul                      |          MatMul ----- w5
#            |                         |            |
#          ReLU                        |           ReLU
#            |                         |            |
#            +------------------------------------ Sum ----- L1Loss
#______________________________________|_______________________________________


class LogChecker(helpers.LogChecker):
    def validate_2x2_parallel_phased_execution(self):
        # pylint: disable=line-too-long
        self.assert_contains("enablePipelining set to value 0")
        self.assert_contains("executionPhaseSettings.stages set to value 2")
        self.assert_contains("executionPhaseSettings.phases set to value 6")
        self.assert_contains(
            "location_activation set to value useOnChipStorage(False)")
        self.assert_contains(
            "location_weight set to value useOnChipStorage(False)")
        self.assert_contains(
            "location_optimizer set to value useOnChipStorage(False)")
        self.assert_contains(
            "location_accumulator set to value useOnChipStorage(False)")

        self.assert_contains(
            "Slice:0 [float32(10, 1), mode(Phased), ipu(0), phase(0)]")
        self.assert_contains(
            "Slice:0/1 [float32(10, 1), mode(Phased), ipu(0), phase(0)]")
        self.assert_contains(
            "MatMul:0 [float32(10, 1), mode(Phased), ipu(0), phase(0)]")
        self.assert_contains(
            "Relu:0 [float32(10, 1), mode(Phased), ipu(0), phase(0)]")

        self.assert_contains(
            "MatMul:0/1 [float32(10, 1), mode(Phased), ipu(2), phase(0)]")
        self.assert_contains(
            "Relu:0/1 [float32(10, 1), mode(Phased), ipu(2), phase(0)]")

        self.assert_contains(
            "MatMul:0/2 [float32(10, 1), mode(Phased), ipu(1), phase(1)]")
        self.assert_contains(
            "Relu:0/2 [float32(10, 1), mode(Phased), ipu(1), phase(1)]")

        self.assert_contains(
            "MatMul:0/3 [float32(10, 1), mode(Phased), ipu(3), phase(1)]")
        self.assert_contains(
            "Relu:0/3 [float32(10, 1), mode(Phased), ipu(3), phase(1)]")

        self.assert_contains(
            "MatMul:0/4 [float32(10, 1), mode(Phased), ipu(1), phase(1)]")
        self.assert_contains(
            "Relu:0/4 [float32(10, 1), mode(Phased), ipu(1), phase(1)]")

        self.assert_contains(
            "MatMul:0/5 [float32(10, 1), mode(Phased), ipu(3), phase(1)]")
        self.assert_contains(
            "Relu:0/5 [float32(10, 1), mode(Phased), ipu(3), phase(1)]")

        self.assert_contains(
            "MatMul:0/6 [float32(10, 1), mode(Phased), ipu(0), phase(2)]")
        self.assert_contains(
            "Relu:0/6 [float32(10, 1), mode(Phased), ipu(0), phase(2)]")

        self.assert_contains(
            "MatMul:0/7 [float32(10, 1), mode(Phased), ipu(2), phase(2)]")
        self.assert_contains(
            "Relu:0/7 [float32(10, 1), mode(Phased), ipu(2), phase(2)]")

        self.assert_contains(
            "MatMul:0/8 [float32(10, 1), mode(Phased), ipu(1), phase(3)]")
        self.assert_contains(
            "Relu:0/8 [float32(10, 1), mode(Phased), ipu(1), phase(3)]")

        self.assert_contains(
            "MatMul:0/9 [float32(10, 1), mode(Phased), ipu(3), phase(3)]")
        self.assert_contains(
            "Relu:0/9 [float32(10, 1), mode(Phased), ipu(3), phase(3)]")

        self.assert_contains(
            "MatMul:0/10 [float32(10, 1), mode(Phased), ipu(1), phase(3)]")
        self.assert_contains(
            "Relu:0/10 [float32(10, 1), mode(Phased), ipu(1), phase(3)]")

        self.assert_contains(
            "MatMul:0/11 [float32(10, 1), mode(Phased), ipu(3), phase(3)]")
        self.assert_contains(
            "Relu:0/11 [float32(10, 1), mode(Phased), ipu(3), phase(3)]")

        self.assert_contains(
            "MatMul:0/12 [float32(10, 1), mode(Phased), ipu(0), phase(4)]")
        self.assert_contains(
            "Relu:0/12 [float32(10, 1), mode(Phased), ipu(0), phase(4)]")

        self.assert_contains(
            "MatMul:0/13 [float32(10, 1), mode(Phased), ipu(2), phase(4)]")
        self.assert_contains(
            "Relu:0/13 [float32(10, 1), mode(Phased), ipu(2), phase(4)]")

        self.assert_contains(
            "MatMul:0/14 [float32(10, 1), mode(Phased), ipu(1), phase(5)]")
        self.assert_contains(
            "Relu:0/14 [float32(10, 1), mode(Phased), ipu(1), phase(5)]")

        self.assert_contains(
            "MatMul:0/15 [float32(10, 1), mode(Phased), ipu(3), phase(5)]")
        self.assert_contains(
            "Relu:0/15 [float32(10, 1), mode(Phased), ipu(3), phase(5)]")

        self.assert_contains(
            "MatMul:0/16 [float32(10, 1), mode(Phased), ipu(1), phase(5)]")
        self.assert_contains(
            "Relu:0/16 [float32(10, 1), mode(Phased), ipu(1), phase(5)]")

        self.assert_contains(
            "MatMul:0/17 [float32(10, 1), mode(Phased), ipu(3), phase(5)]")
        self.assert_contains(
            "Relu:0/17 [float32(10, 1), mode(Phased), ipu(3), phase(5)]")
        self.assert_contains(
            "Add:0 [float32(10, 1), mode(Phased), ipu(3), phase(5)]")
        self.assert_contains(
            "Sub:0 [float32(10, 1), mode(Phased), ipu(3), phase(5)]")
        self.assert_contains(
            "L1:0 [float32(shape inference failed), mode(Phased), ipu(3), phase(5)]"
        )
        self.assert_contains(
            "IdentityLoss:0 [float32(shape inference failed), mode(Phased), ipu(3), phase(5)]"
        )
        # pylint: enable=line-too-long

    def validate_2x2_parallel_phased_execution_small(self):
        # pylint: disable=line-too-long
        self.assert_contains("enablePipelining set to value 0")
        self.assert_contains("executionPhaseSettings.stages set to value 2")
        self.assert_contains("executionPhaseSettings.phases set to value 2")
        self.assert_contains(
            "location_activation set to value useOnChipStorage(False)")
        self.assert_contains(
            "location_weight set to value useOnChipStorage(False)")
        self.assert_contains(
            "location_optimizer set to value useOnChipStorage(False)")
        self.assert_contains(
            "location_accumulator set to value useOnChipStorage(False)")

        self.assert_contains(
            "Slice:0 [float32(10, 1), mode(Phased), ipu(0), phase(0)]")
        self.assert_contains(
            "Slice:0/1 [float32(10, 1), mode(Phased), ipu(0), phase(0)]")
        self.assert_contains(
            "MatMul:0 [float32(10, 1), mode(Phased), ipu(0), phase(0)]")
        self.assert_contains(
            "Relu:0 [float32(10, 1), mode(Phased), ipu(0), phase(0)]")

        self.assert_contains(
            "MatMul:0/1 [float32(10, 1), mode(Phased), ipu(2), phase(0)]")
        self.assert_contains(
            "Relu:0/1 [float32(10, 1), mode(Phased), ipu(2), phase(0)]")

        self.assert_contains(
            "MatMul:0/2 [float32(10, 1), mode(Phased), ipu(1), phase(1)]")
        self.assert_contains(
            "Relu:0/2 [float32(10, 1), mode(Phased), ipu(1), phase(1)]")

        self.assert_contains(
            "MatMul:0/3 [float32(10, 1), mode(Phased), ipu(3), phase(1)]")
        self.assert_contains(
            "Relu:0/3 [float32(10, 1), mode(Phased), ipu(3), phase(1)]")

        self.assert_contains(
            "MatMul:0/4 [float32(10, 1), mode(Phased), ipu(1), phase(1)]")
        self.assert_contains(
            "Relu:0/4 [float32(10, 1), mode(Phased), ipu(1), phase(1)]")

        self.assert_contains(
            "MatMul:0/5 [float32(10, 1), mode(Phased), ipu(3), phase(1)]")
        self.assert_contains(
            "Relu:0/5 [float32(10, 1), mode(Phased), ipu(3), phase(1)]")

        self.assert_contains(
            "Add:0 [float32(10, 1), mode(Phased), ipu(3), phase(1)]")
        self.assert_contains(
            "Sub:0 [float32(10, 1), mode(Phased), ipu(3), phase(1)]")
        self.assert_contains(
            "L1:0 [float32(shape inference failed), mode(Phased), ipu(3), phase(1)]"
        )
        self.assert_contains(
            "IdentityLoss:0 [float32(shape inference failed), mode(Phased), ipu(3), phase(1)]"
        )
        # pylint: enable=line-too-long

    def validate_serial_tensor_liveness(self, liveness):
        # 'phases' does not include the bwd pass, so to calculate,
        # sum the number of phases in the fwd pass, plus any phase
        # gap between the end of the fwd and start of the bwd pass
        if liveness == poptorch.Liveness.AlwaysLive:
            # fwd:       bwd:
            # phase 0 -> phase 4
            # phase 1 -> phase 3
            # phase 2 -> phase 2
            phases = 3
            stride = 1
        elif liveness == poptorch.Liveness.OffChipAfterFwd:
            # fwd:       bwd:
            # phase 0 -> phase 8
            # phase 1 -> phase 7
            # phase 2 -> phase 6
            phases = 6
            stride = 1
        elif liveness == poptorch.Liveness.OffChipAfterFwdNoOverlap:
            # fwd:       bwd:
            # phase 0 -> phase 12
            # phase 2 -> phase 10
            # phase 4 -> phase 8
            phases = 8
            stride = 2
        elif liveness == poptorch.Liveness.OffChipAfterEachPhase:
            # fwd:       bwd:
            # phase 0 -> phase 20
            # phase 4 -> phase 16
            # phase 8 -> phase 12
            phases = 12
            stride = 4

        self.assert_contains('set serial_phases_execution to value true')
        self.assert_contains('executionPhaseSettings.stages set to value 1')

        self.assert_contains(
            'executionPhaseSettings.phases set to value {}'.format(phases))

        for phase in range(3):
            op_label = ':0'
            self.assert_contains(
                'Transpose{} [float32({}, {}), mode(Phased), ipu(0), phase({})]'
                .format(op_label, 8 - phase, 7 - phase, phase * stride))
            self.assert_contains(
                'MatMul{} [float32({}), mode(Phased), ipu(0), phase({})]'.
                format(op_label, 7 - phase, phase * stride))
            self.assert_contains(
                'Add{} [float32({}), mode(Phased), ipu(0), phase({})]'.format(
                    op_label, 7 - phase, phase * stride))


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
@pytest.mark.parametrize("trace_model", [True, False])
def test_2x2_parallel_phased_execution_inline(capfd, trace_model):
    N = 3
    size = 10

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.ParameterList([
                torch.nn.Parameter(torch.rand(size, size), requires_grad=True)
                for n in range(N * 6)
            ])

        def forward(self, in0, target=None):
            phase = 0
            with poptorch.Block("0", ipu_id=0):
                ins = torch.split(in0, size)
            weight = iter(self.weights)
            for n in range(N * 3):
                out = []
                for ipu in range(2):
                    x = ins[ipu]
                    # Alternate between 0-2 and 1-3
                    ipu = (phase % 2) + ipu * 2
                    with poptorch.Block(f"{phase}", ipu_id=ipu):
                        x = torch.matmul(next(weight), x)
                        out.append(F.relu(x))
                ins = out[1], out[0]
                # We want 2 matmuls in the same phase
                if n % 3 != 1:
                    phase += 1
            with poptorch.Block(f"{N*2-1}", ipu_id=3):
                res = ins[0] + ins[1]
                if target is None:
                    return res
                return res, torch.nn.L1Loss(reduction="mean")(res, target)

    input = torch.rand(size * 2, 1)
    target = torch.rand(size, 1)

    model = Model()
    opts = poptorch.Options()
    opts.Jit.traceModel(trace_model)

    phases = []
    phases = [f"{n}" for n in range(2 * N)]
    opts.setExecutionStrategy(poptorch.ParallelPhasedExecution(*phases))
    poptorch_model = poptorch.trainingModel(model, opts)
    poptorch_model.compile(input, target)

    testlog = LogChecker(capfd)
    testlog.validate_2x2_parallel_phased_execution()


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
@pytest.mark.parametrize("trace_model", [True, False])
def test_2x2_parallel_phased_execution_opts(capfd, trace_model):
    N = 3
    size = 10

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.ParameterList([
                torch.nn.Parameter(torch.rand(size, size), requires_grad=True)
                for n in range(N * 6)
            ])

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
    opts.Jit.traceModel(trace_model)
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

    testlog = LogChecker(capfd)
    testlog.validate_2x2_parallel_phased_execution()


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
@pytest.mark.parametrize("trace_model", [True, False])
def test_2x2_parallel_phased_execution_small_opts(capfd, trace_model):
    size = 10

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.ParameterList([
                torch.nn.Parameter(torch.rand(size, size), requires_grad=True)
                for n in range(6)
            ])

        def forward(self, in0, target=None):
            poptorch.Block.useAutoId()
            weight = iter(self.weights)

            # Phase 0 / ipu 0
            with poptorch.Block():
                in0, in1 = torch.split(in0, size)
                x = torch.matmul(next(weight), in0)
                out0 = F.relu(x)

            # Phase 0 / ipu 2
            with poptorch.Block():
                x = torch.matmul(next(weight), in1)
                out1 = F.relu(x)

            in0, in1 = out1, out0

            # Phase 1 / ipu 1
            with poptorch.Block():
                x = torch.matmul(next(weight), in0)
                out0 = F.relu(x)

            # Phase 1 / ipu 3
            with poptorch.Block():
                x = torch.matmul(next(weight), in1)
                out1 = F.relu(x)

            in0, in1 = out1, out0

            # Phase 1 / ipu 1 - part 2
            with poptorch.Block():
                x = torch.matmul(next(weight), in0)
                out0 = F.relu(x)

            # Phase 1 / ipu 3 - part 2
            with poptorch.Block():
                x = torch.matmul(next(weight), in1)
                out1 = F.relu(x)
                res = out0 + out1
                if target is None:
                    return res
                return res, torch.nn.L1Loss(reduction="mean")(res, target)

    input = torch.rand(size * 2, 1)
    target = torch.rand(size, 1)
    model = Model()
    opts = poptorch.Options()
    opts.Jit.traceModel(trace_model)
    strategy = poptorch.ParallelPhasedExecution(
        [poptorch.Stage("0"), poptorch.Stage("1")],
        [poptorch.Stage("2", "4"),
         poptorch.Stage("3", "5")])
    # Alternate between 0-2 and 1-3
    strategy.phase(0).ipus(0, 2)
    strategy.phase(1).ipus(1, 3)

    opts.setExecutionStrategy(strategy)
    poptorch_model = poptorch.trainingModel(model, opts)
    poptorch_model.compile(input, target)

    testlog = LogChecker(capfd)
    testlog.validate_2x2_parallel_phased_execution_small()


@pytest.mark.parametrize("liveness", list(poptorch.Liveness))
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
@pytest.mark.parametrize("trace_model", [True, False])
def test_serial_tensor_liveness(capfd, liveness, trace_model):
    if not trace_model:
        pytest.skip("TODO(T57195): AssertionError")

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = torch.nn.Linear(8, 7)
            self.fc2 = torch.nn.Linear(7, 6)
            self.fc3 = torch.nn.Linear(6, 5)

        def forward(self, x):
            with poptorch.Block("B1"):
                x = self.fc1(x)
            with poptorch.Block("B2"):
                x = self.fc2(x)
            with poptorch.Block("B3"):
                x = self.fc3(x)
            return x

    strategy = poptorch.SerialPhasedExecution("B1", "B2", "B3")
    strategy.stage("B1").ipu(0)
    strategy.stage("B2").ipu(0)
    strategy.stage("B3").ipu(0)
    strategy.setTensorsLiveness(liveness)
    opts = poptorch.Options()
    opts.setExecutionStrategy(strategy)
    opts.Jit.traceModel(trace_model)

    model = Model()
    model = poptorch.inferenceModel(model, opts)

    input = torch.randn(8)
    model.compile(input)

    testlog = LogChecker(capfd)
    testlog.validate_serial_tensor_liveness(liveness)


def test_phased_api():
    # Try to pass a list of Phases
    poptorch.SerialPhasedExecution(
        poptorch.Phase('layer1'),
        poptorch.Phase('layer2'),
    )

    # Try to pass a list of stages
    poptorch.SerialPhasedExecution(
        poptorch.Stage('layer1'),
        poptorch.Stage('layer2'),
    )

    # Try to pass a list of list of stages
    poptorch.SerialPhasedExecution(
        [poptorch.Stage('layer1'),
         poptorch.Stage('layer1.b')],
        [poptorch.Stage('layer2'),
         poptorch.Stage('layer2.b')])

    # Try to pass a list of list of block IDs
    poptorch.SerialPhasedExecution(["layer1"], ["layer2"])
