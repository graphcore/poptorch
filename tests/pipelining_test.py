#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import json
import torch
import pytest
import helpers
import poptorch


def test_missing_block():
    poptorch.setLogLevel(1)  # Force debug logging

    class Model(torch.nn.Module):
        def forward(self, x):
            poptorch.Block.useAutoId()
            with poptorch.Block(ipu_id=0):
                x = x * 4
            x = x * 4
            return x

    m = Model()

    opts = poptorch.Options()
    opts.deviceIterations(2)
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))

    m = poptorch.inferenceModel(m, opts)
    with pytest.raises(RuntimeError, match="No active Block"):
        m.compile(torch.randn(2, 5))


@helpers.printCapfdOnExit
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
    opts.deviceIterations(2)

    m = poptorch.inferenceModel(m, opts)
    m(torch.randn(2, 5))

    log = helpers.LogChecker(capfd)
    log.assert_contains("enablePipelining set to value 1")
    log.assert_contains(" Mul:0 ", " mode(Pipelined), ipu(0), stage(0)")
    log.assert_contains(" Mul:0/1 ", " mode(Pipelined), ipu(1), stage(1)")


def run_recomputation_checkpoint_test(size, model_cls, exp_num_stash_ckpted):
    # pylint: disable=protected-access
    poptorch.setLogLevel(1)  # Force debug logging

    dev_its = 6

    opts = poptorch.Options()
    opts.deviceIterations(dev_its)
    opts._Popart.set("autoRecomputation", 3)  # All forward pipeline stages.

    m = poptorch.trainingModel(model_cls(False), opts)
    m.compile(torch.randn(dev_its, size, 1), torch.randn(dev_its, size, 1))
    ir = json.loads(m._debugGetPopartIR())
    assert not any(["Checkpoint" in node["name"] for node in ir["maingraph"]
                    ]), ("Popart IR shouldn't contain any checkpoint")
    assert sum(["Stash" in node["type"] for node in ir["maingraph"]
                ]) == 1, ("Only the graph input should be stashed")

    native_ckpted = model_cls(True)
    m = poptorch.trainingModel(native_ckpted, opts)
    m.compile(torch.randn(dev_its, size, 1), torch.randn(dev_its, size, 1))
    ir = json.loads(m._debugGetPopartIR())  # pylint: disable=protected-access
    assert any(["Checkpoint" in node["name"] for node in ir["maingraph"]
                ]), ("Popart IR should contain a checkpoint")
    assert sum([
        "Stash" in node["type"] for node in ir["maingraph"]
    ]) == exp_num_stash_ckpted, ("Both the graph input and the checkpoint(s) "
                                 "should be stashed")


def test_recomputation_checkpoint_tensor():
    size = 3

    class Model(torch.nn.Module):
        def __init__(self, checkpoint=False):
            super().__init__()
            self.checkpoint = checkpoint
            weight = torch.nn.Parameter(torch.rand(size, size),
                                        requires_grad=True)
            self.register_parameter("weight", weight)

        def forward(self, x, target):
            poptorch.Block.useAutoId()
            with poptorch.Block(ipu_id=0):
                x = torch.matmul(self.weight, x)
                if self.checkpoint:
                    x = poptorch.recomputationCheckpoint(x)
                x = torch.matmul(self.weight, x)

            with poptorch.Block(ipu_id=1):
                x = x * 2
                return x, torch.nn.functional.l1_loss(x, target)

    run_recomputation_checkpoint_test(size, Model, 2)


def test_recomputation_checkpoint_tensor_two_inputs():
    size = 3

    class Model(torch.nn.Module):
        def __init__(self, checkpoint=False):
            super().__init__()
            self.checkpoint = checkpoint
            weight_1 = torch.nn.Parameter(torch.rand(size, size),
                                          requires_grad=True)
            self.register_parameter("weight_1", weight_1)

            weight_2 = torch.nn.Parameter(torch.rand(size, size),
                                          requires_grad=True)
            self.register_parameter("weight_2", weight_2)

        def forward(self, x, target):
            poptorch.Block.useAutoId()
            with poptorch.Block(ipu_id=0):
                x = torch.matmul(self.weight_1, x)
                y = torch.matmul(self.weight_2, x)

                if self.checkpoint:
                    x, y = poptorch.recomputationCheckpoint(x, y)
                x = torch.matmul(self.weight_1, x + y)

            with poptorch.Block(ipu_id=1):
                x = x * 2
                return x, torch.nn.functional.l1_loss(x, target)

    run_recomputation_checkpoint_test(size, Model, 3)


def test_recomputation_checkpoint_tensor_tuple_inputs():
    size = 3

    class Model(torch.nn.Module):
        def __init__(self, checkpoint=False):
            super().__init__()
            self.checkpoint = checkpoint
            weight_1 = torch.nn.Parameter(torch.rand(size, size),
                                          requires_grad=True)
            self.register_parameter("weight_1", weight_1)

            weight_2 = torch.nn.Parameter(torch.rand(size, size),
                                          requires_grad=True)
            self.register_parameter("weight_2", weight_2)

        def forward(self, x, target):
            poptorch.Block.useAutoId()
            with poptorch.Block(ipu_id=0):
                x = torch.matmul(self.weight_1, x)
                y = torch.matmul(self.weight_2, x)

                if self.checkpoint:
                    x, y = poptorch.recomputationCheckpoint((x, y))
                x = torch.matmul(self.weight_1, x + y)

            with poptorch.Block(ipu_id=1):
                x = x * 2
                return x, torch.nn.functional.l1_loss(x, target)

    run_recomputation_checkpoint_test(size, Model, 3)


@helpers.printCapfdOnExit
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
    poptorch.BeginBlock(m.l1, ipu_id=0)
    poptorch.BeginBlock(m.l2, ipu_id=0)

    opts = poptorch.Options()
    opts.deviceIterations(2)

    m = poptorch.inferenceModel(m, opts)
    m(torch.randn(2, 5))

    log = helpers.LogChecker(capfd)
    log.assert_contains("enablePipelining set to value 0")
    log.assert_contains(" Mul:0 ", " mode(Pipelined), ipu(0), stage(0)")
    log.assert_contains(" Mul:0/1 ", " mode(Pipelined), ipu(0), stage(0)")


@helpers.printCapfdOnExit
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
    poptorch.BeginBlock(m.l1, ipu_id=1)
    poptorch.BeginBlock(m.l2, ipu_id=1)

    opts = poptorch.Options()
    opts.deviceIterations(2)

    m = poptorch.inferenceModel(m, opts)
    m(torch.randn(2, 5))

    log = helpers.LogChecker(capfd)
    log.assert_contains("enablePipelining set to value 1")
    log.assert_contains(" Mul:0 ", " mode(Pipelined), ipu(0), stage(0)")
    log.assert_contains(" Mul:0/1 ", " mode(Pipelined), ipu(1), stage(1)")
    log.assert_contains(" Mul:0/2 ", " mode(Pipelined), ipu(1), stage(1)")


@helpers.printCapfdOnExit
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
    opts.deviceIterations(4).autoRoundNumIPUs(True)
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


@helpers.printCapfdOnExit
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
    opts.deviceIterations(4).autoRoundNumIPUs(True)
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


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Round up only needed for IPU Hardware")
def test_ipu_round_up_error():
    class Block(torch.nn.Module):
        def forward(self, x):
            return x * 6

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = Block()
            self.l2 = Block()
            self.l3 = Block()

        def forward(self, x):
            x = self.l1(x)
            x = self.l2(x)
            x = self.l3(x)
            return x

    m = Model()
    poptorch.BeginBlock(m.l1, ipu_id=0)
    poptorch.BeginBlock(m.l2, ipu_id=1)
    poptorch.BeginBlock(m.l3, ipu_id=2)

    opts = poptorch.Options()
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))

    m = poptorch.inferenceModel(m, opts)

    error_msg = (
        ".+The model specifies the use of 3 IPUs, however PopTorch must "
        "reserve a minimum of 4 in order to allow the model to run, "
        "because PopTorch must reserve a power of 2 or maximum of 64 IPUs per "
        r"process\. Please reconfigure your model to use a different number of "
        r"IPUs or set poptorch\.Options\(\)\.autoRoundNumIPUs\(True\)\.")
    with pytest.raises(RuntimeError, match=error_msg):
        m(torch.randn(4, 5))


def test_begin_block_names_invariant():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

            self.l1 = torch.nn.Linear(3, 5)
            self.l2 = torch.nn.Linear(5, 5)
            self.l3 = torch.nn.Linear(5, 3)

        def forward(self, x):
            x = self.relu(self.l1(x))
            x = self.relu(self.l2(x))
            x = self.relu(self.l3(x))
            return x

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = Block()
            self.l2 = Block()

        def forward(self, x):
            x = self.l1(x)
            with poptorch.Block(ipu_id=2):
                x = self.l2(x)
            return x

    m = Model()

    old_all_names = [n for n, _ in m.named_parameters()]
    old_state_dict = m.state_dict()

    m_l1_wrapped = poptorch.BeginBlock(m.l1, ipu_id=1)

    # The return is for backward compatibility
    assert m_l1_wrapped is m.l1

    print(m.l2.__class__)
    assert m.l2.__class__ is Block
    poptorch.BeginBlock(m.l2, ipu_id=2)
    assert m.l2.__class__ is not Block

    new_all_names = [n for n, _ in m.named_parameters()]
    new_state_dict = m.state_dict()

    assert old_all_names == new_all_names

    sorted_state_dict_keys = sorted(old_state_dict.keys())
    assert sorted_state_dict_keys == sorted(new_state_dict.keys())

    for k in sorted_state_dict_keys:
        helpers.assert_allequal(expected=old_state_dict[k],
                                actual=new_state_dict[k])

    # Strict=True is a sufficient test in itself
    m.load_state_dict(old_state_dict, strict=True)

    # Test dir does not raise an exception
    dir(m.l1)

    # Test registering a buffer
    m.l1.register_buffer("a_buff",
                         torch.nn.parameter.Parameter(torch.zeros(2, 2)))

    buffer_names = [b[0] for b in m.named_buffers()]
    assert "l1.a_buff" in buffer_names

    # Test registering a param
    m.l1.register_parameter("a_param",
                            torch.nn.parameter.Parameter(torch.zeros(2, 2)))

    param_names = [p[0] for p in m.named_parameters()]
    assert "l1.a_param" in param_names
