#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import pytest
import helpers
import poptorch


@pytest.mark.parametrize("trace_model", [True, False])
def test_inferenceBatching(trace_model):
    torch.manual_seed(42)

    model = torch.nn.Linear(6, 20)

    # Actually batched by 100.
    input = torch.randn([10, 1, 5, 6])

    # Run pytorch native on CPU batchsize 10.
    native_output = model(input)

    # Run on IPU batch size 1 * 10 popart batches.
    opts = poptorch.Options().deviceIterations(10)
    opts.Jit.traceModel(trace_model)
    ipuModel = poptorch.inferenceModel(model, opts)
    poptorch_out = ipuModel(input)

    # Check that inference wrapper has defaulted to "All".
    assert len(poptorch_out.size()) == 4
    assert poptorch_out.size()[0] == 10
    helpers.assert_allclose(expected=native_output, actual=poptorch_out)


@pytest.mark.parametrize("trace_model", [True, False])
def test_trainingBatching(trace_model):
    torch.manual_seed(4424242)

    # 10 Batches of 10.
    input = torch.randn(10, 10)

    # 10 batches of 1
    label = torch.randint(0, 10, [1])
    label = label.expand([10])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, data, target):
            out = self.linear(data)
            loss = self.loss(out, target)
            return out, loss

    model = Model()

    # Run on IPU batch size 1 * 10 popart batches.
    opts = poptorch.Options().deviceIterations(10)
    opts.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=opts)

    # Run all 10 batches as batchsize 10.
    out, _ = model(input, label)

    # Sanity check we weren't already matching the label.
    assert not torch.equal(torch.argmax(out, dim=1), label)

    for _ in range(0, 1000):
        _, loss = poptorch_model(input, label)

        # Each batch should NOT report its own loss. As by default training model should have a "Final" output mode.
        assert len(loss.size()) == 0

    # Run with trained weights.
    out, _ = model(input, label)

    # Check we are now equal with labels.
    helpers.assert_allequal(actual=torch.argmax(out, dim=1), expected=label)


@pytest.mark.parametrize("mode", list(poptorch.OutputMode))
@pytest.mark.parametrize("trace_model", [True, False])
def test_inferenceOutputModes(mode, trace_model):
    torch.manual_seed(42)

    model = torch.nn.Linear(6, 20)

    # Actually batched by 100.
    input = torch.randn([10, 1, 5, 6])

    # Run pytorch native on CPU batchsize 10.
    native_out = model(input)

    # Run on IPU batch size 1 * 10 popart batches. output_return_period ignored if not EVERYN
    opts = poptorch.Options().deviceIterations(10)
    opts.outputMode(mode, output_return_period=5)
    opts.Jit.traceModel(trace_model)
    ipuModel = poptorch.inferenceModel(model, opts)
    poptorch_out = ipuModel(input)

    if mode in [poptorch.OutputMode.All, poptorch.OutputMode.Default]:
        # Expect the full batch.
        assert len(poptorch_out.size()) == 4
        assert poptorch_out.size()[0] == 10
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)
    elif mode == poptorch.OutputMode.EveryN:
        # Otherwise we are expecting device_iterations / N
        assert len(poptorch_out.size()) == 4
        assert poptorch_out.size()[0] == 2

        # Check each N is the correct batch
        helpers.assert_allclose(actual=poptorch_out[0], expected=native_out[4])
        helpers.assert_allclose(actual=poptorch_out[1], expected=native_out[9])

    else:
        # Otherwise we are expecting just one element per batch.
        assert len(poptorch_out.size()) == 4
        assert poptorch_out.size()[0] == 1

        if mode == poptorch.OutputMode.Final:
            # Check we are the same as the last output.
            helpers.assert_allclose(actual=poptorch_out.reshape(
                native_out[-1].shape),
                                    expected=native_out[-1])
        elif mode == poptorch.OutputMode.Sum:
            # Check we are close to the sum of the batch dim.
            sum = torch.sum(native_out, dim=0, keepdim=True)
            helpers.assert_allclose(actual=poptorch_out, expected=sum)
        else:
            assert False, "Unexpected output mode %s" % mode


@pytest.mark.parametrize("mode", list(poptorch.OutputMode))
@pytest.mark.parametrize("trace_model", [True, False])
def test_trainingOutputModes(mode, trace_model):
    torch.manual_seed(42)

    # 1000 Batches of 10.
    input = torch.randn(1000, 10)

    # 1000 batches of 1
    label = torch.randint(0, 10, [1])
    label = label.expand([1000])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, data, target):
            out = self.linear(data)
            loss = self.loss(out, target)
            return out, loss

    model = Model()

    # Run pytorch native on CPU batchsize 10.
    model(input, label)

    # Run on IPU batch size 1 * 1000 popart batches.
    opts = poptorch.Options().deviceIterations(1000)
    opts.outputMode(mode, output_return_period=20)
    opts.Jit.traceModel(trace_model)

    poptorch_model = poptorch.trainingModel(model, options=opts)

    poptorch_out, loss = poptorch_model(input, label)

    if mode == poptorch.OutputMode.All:
        # Expect the full batch.
        assert len(poptorch_out.size()) == 2
        assert poptorch_out.size()[0] == 1000

        assert len(loss.size()) == 1
        assert loss.size()[0] == 1000

        # Check the rolling average loss is downward sloped.
        interval = 100
        previous_average = torch.mean(loss[:interval])
        for i in range(1, 1000 // interval):
            start = interval * i
            end = start + interval
            new_average = torch.mean(loss[start:end])

            assert new_average < previous_average

            previous_average = new_average

    elif mode == poptorch.OutputMode.EveryN:
        # Otherwise we are expecting device_iterations / N
        assert len(poptorch_out.size()) == 2
        assert poptorch_out.size()[0] == 50

        # There's too much noise in the losses for us to test directly without averaging like above so just test sizes.
        assert len(loss.size()) == 1
        assert loss.size()[0] == 50
    else:
        # Otherwise we are expecting just one element per batch.
        assert len(poptorch_out.size()) == 2
        assert poptorch_out.size()[0] == 1

        assert len(loss.size()) == 0

        if mode in [poptorch.OutputMode.Final, poptorch.OutputMode.Default]:
            # We just have to check the loss is small.
            # This is just relative to the previously observed loss values on this test with this seed.
            assert loss < 0.2

        elif mode == poptorch.OutputMode.Sum:
            # We just have to check that the loss is huge.
            assert loss > 500.0
        else:
            assert False, "Unexpected output mode %s" % mode


def run_gradient_accumulation_test(input, target, gradient_accumulations,
                                   accumulation_reduction_type, lr,
                                   trace_model):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.loss = torch.nn.L1Loss(reduction="mean")

        def forward(self, data, target):
            out = self.linear(data)
            loss = self.loss(out, target)
            return out, loss

    model = Model()

    opts = poptorch.Options()
    opts.outputMode(poptorch.OutputMode.All)
    opts.Training.gradientAccumulation(gradient_accumulations)
    opts.Jit.traceModel(trace_model)

    if accumulation_reduction_type is not None:
        opts.Training.accumulationAndReplicationReductionType(
            accumulation_reduction_type)

    poptorch_model = poptorch.trainingModel(model,
                                            options=opts,
                                            optimizer=torch.optim.SGD(
                                                model.parameters(), lr=lr))

    # Run 10 training steps
    for _ in range(10):
        poptorch_model(input, target)

    # return trained weight matrix
    return poptorch_model.linear.weight.data


@pytest.mark.parametrize("trace_model", [True, False])
def test_gradient_accumulation_training(trace_model):
    torch.manual_seed(42)

    target = torch.randn(4, 10)
    input = torch.randn(4, 10)

    # Testing gradient accumulations 1 vs 2 and Mean reduction
    w_with_1 = run_gradient_accumulation_test(target, input, 1,
                                              poptorch.ReductionType.Mean,
                                              0.01, trace_model)
    w_with_2 = run_gradient_accumulation_test(target, input, 2,
                                              poptorch.ReductionType.Mean,
                                              0.01, trace_model)
    helpers.assert_allclose(actual=w_with_1, expected=w_with_2)

    # Test the default matches as well (i.e. the default is mean)
    w_with_2 = run_gradient_accumulation_test(target, input, 2, None, 0.01,
                                              trace_model)
    helpers.assert_allclose(actual=w_with_1, expected=w_with_2)

    # Testing gradient accumulations 1 vs 2 and Sum reduction (different lr)
    w_with_1 = run_gradient_accumulation_test(target, input, 1,
                                              poptorch.ReductionType.Sum, 0.02,
                                              trace_model)
    w_with_2 = run_gradient_accumulation_test(target, input, 2,
                                              poptorch.ReductionType.Sum, 0.01,
                                              trace_model)
    helpers.assert_allclose(actual=w_with_1, expected=w_with_2)


class FourBlockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(1, 1)
        self.lin2 = torch.nn.Linear(1, 1)
        self.lin3 = torch.nn.Linear(1, 1)
        self.lin4 = torch.nn.Linear(1, 1)

    def forward(self, x):
        with poptorch.Block("B1", ipu_id=0):
            out = self.lin1(x)
        with poptorch.Block("B2", ipu_id=1):
            out = self.lin2(out)
        with poptorch.Block("B3", ipu_id=2):
            out = self.lin3(out)
        with poptorch.Block("B4", ipu_id=3):
            out = self.lin4(out)

        return out


class FourBlockModelNoScope(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(1, 1)
        self.lin2 = torch.nn.Linear(1, 1)
        self.lin3 = torch.nn.Linear(1, 1)
        self.lin4 = torch.nn.Linear(1, 1)

    def forward(self, x):
        poptorch.Block.start("B1", ipu_id=0)
        out = self.lin1(x)
        poptorch.Block.start("B2", ipu_id=1)
        out = self.lin2(out)
        poptorch.Block.start("B3", ipu_id=2)
        out = self.lin3(out)
        poptorch.Block.start("B4", ipu_id=3)
        out = self.lin4(out)

        return out


@pytest.mark.parametrize("num_grad_accums", (4, 5, 7))
@pytest.mark.parametrize("device_iterations", (1, 2))
@pytest.mark.parametrize("trace_model", [True, False])
def test_gradient_accumulation_pipelined_training(num_grad_accums,
                                                  device_iterations,
                                                  trace_model):
    class TrainingFourBlockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.four_block = FourBlockModel()

        def forward(self, x):
            out = self.four_block(x)
            with poptorch.Block("B4", ipu_id=3):
                loss = poptorch.identity_loss(out, reduction="mean")

            return out, loss

    model = TrainingFourBlockModel()
    opts = poptorch.Options()
    opts.deviceIterations(device_iterations)
    opts.Training.gradientAccumulation(num_grad_accums)
    opts.Jit.traceModel(trace_model)

    poptorch_model = poptorch.trainingModel(model, options=opts)

    if num_grad_accums in (4, 5):
        err_msg = (r"poptorch\.Options\(\)\.Training\.gradientAccumulation "
                   r"must be greater than or equal to the number of pipeline"
                   r" stages \(7\) when using poptorch\.PipelinedExecution\. "
                   r"Please note that a model with 4 pipeline stages in "
                   r"PopTorch will have an additional 3 stages when training.")

        with pytest.raises(poptorch.poptorch_core.Error, match=err_msg):
            poptorch_model(torch.zeros(num_grad_accums * device_iterations))
    else:
        poptorch_model(torch.zeros(num_grad_accums * device_iterations))


@pytest.mark.parametrize("pipelined", [True, False])
@pytest.mark.parametrize("Model", [FourBlockModel, FourBlockModelNoScope])
def test_gradient_accumulation_inference(pipelined, Model):
    model = Model()
    opts = poptorch.Options()

    if pipelined:
        # pylint: disable=protected-access
        assert isinstance(opts._execution_strategy,
                          poptorch.PipelinedExecution)
    else:
        opts.setExecutionStrategy(poptorch.ShardedExecution())

    opts.Training.gradientAccumulation(2)

    err_msg = (r"You must set "
               r"poptorch\.Options\(\)\.Training\.gradientAccumulation to 1 "
               r"or leave it as its default value \(1\) when running a "
               r"poptorch\.inferenceModel\(\)\.")

    if pipelined:
        err_msg += (r" Use poptorch\.Options\(\)\.deviceIterations() to "
                    r"process a sufficient number of batches each run for "
                    r"pipelined execution instead.")

    with pytest.raises(poptorch.poptorch_core.Error, match=err_msg):
        poptorch.inferenceModel(model, options=opts)


@pytest.mark.parametrize("pipelined", [True, False])
@pytest.mark.parametrize("device_iterations", (2, 4))
@pytest.mark.parametrize("Model", [FourBlockModel, FourBlockModelNoScope])
def test_device_iterations_inference(pipelined, device_iterations, Model):
    model = Model()
    opts = poptorch.Options()

    if pipelined:
        # pylint: disable=protected-access
        assert isinstance(opts._execution_strategy,
                          poptorch.PipelinedExecution)
    else:
        opts.setExecutionStrategy(poptorch.ShardedExecution())

    opts.deviceIterations(device_iterations)

    poptorch_model = poptorch.inferenceModel(model, options=opts)

    if pipelined and device_iterations == 2:
        err_msg = (r"poptorch\.Options\(\)\.deviceIterations must be greater")
        with pytest.raises(poptorch.poptorch_core.Error, match=err_msg):
            poptorch_model(torch.zeros(device_iterations))
    else:
        poptorch_model(torch.zeros(device_iterations))
