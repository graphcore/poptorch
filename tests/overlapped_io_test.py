#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
import pytest

import poptorch

INPUT_SIZE = 64


def get_model(num_mat_muls,
              input_a_overlap=poptorch.OverlapMode.NoOverlap,
              input_b_overlap=poptorch.OverlapMode.NoOverlap):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

            for idx in range(num_mat_muls):
                self.register_parameter(
                    "a" + str(idx),
                    torch.nn.Parameter(
                        torch.randn([1, INPUT_SIZE, INPUT_SIZE],
                                    dtype=torch.float32)))
                self.register_parameter(
                    "b" + str(idx),
                    torch.nn.Parameter(
                        torch.randn([1, INPUT_SIZE, INPUT_SIZE],
                                    dtype=torch.float32)))

            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, input_a, input_b, labels):
            with poptorch.Block(ipu_id=0):
                # Set overlap settings
                input_a = poptorch.set_overlap_for_input(
                    input_a, input_a_overlap)

                input_b = poptorch.set_overlap_for_input(
                    input_b, input_b_overlap)

                # remove leading 1 dim
                input_a = input_a.squeeze()
                input_b = input_b.squeeze()

                to_sum = []

                for idx in range(num_mat_muls):
                    to_sum.append(
                        torch.matmul(self.get_parameter("a" + str(idx)),
                                     input_a))
                    to_sum.append(
                        torch.matmul(self.get_parameter("b" + str(idx)),
                                     input_b))

                sum_all = torch.sum(torch.stack(to_sum, dim=0), dim=0)
                print(sum_all.shape)

                loss = self.loss(sum_all.unsqueeze(dim=0), labels)

                return loss, sum_all

    return Model()


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_io_input():
    num_mat_muls = 20
    model = get_model(num_mat_muls,
                      poptorch.OverlapMode.OverlapAccumulationLoop,
                      poptorch.OverlapMode.OverlapAccumulationLoop)
    num_grad_accumulations = 10
    num_device_iterations = 20

    opts = poptorch.Options()
    opts.anchorMode(poptorch.AnchorMode.All)
    opts.deviceIterations(num_device_iterations)
    opts.setExecutionStrategy(poptorch.ShardedExecution())

    opts.TensorLocations.numIOTiles(32)

    opts.Training.gradientAccumulation(num_grad_accumulations)
    poptorch_model = poptorch.trainingModel(model, options=opts)

    total_batch_size = num_grad_accumulations * num_device_iterations

    input_a = torch.randn((total_batch_size, INPUT_SIZE))
    input_b = torch.randn((total_batch_size, INPUT_SIZE))
    labels = torch.randint(0, 1, (total_batch_size, INPUT_SIZE))

    poptorch_model(input_a, input_b, labels)


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_input_error_messages():
    class DoubleInputUseModel(torch.nn.Module):
        def forward(self, x):
            y = x + 1
            x2 = poptorch.set_overlap_for_input(
                x, poptorch.OverlapMode.OverlapAccumulationLoop)
            return y, x2

    model = DoubleInputUseModel()
    poptorch_model = poptorch.inferenceModel(model)

    err_msg = (
        r"poptorch\.set_overlap_for_input must be the only op applied " +
        r"to an input\. This is not the case for input x to the model\.")
    with pytest.raises(poptorch.poptorch_core.Error, match=err_msg):
        poptorch_model(torch.tensor([1.0]))

    class NotOnInputModel(torch.nn.Module):
        def forward(self, x):
            y = x + 1
            y2 = poptorch.set_overlap_for_input(
                y, poptorch.OverlapMode.OverlapAccumulationLoop)
            return y, y2

    model = NotOnInputModel()
    poptorch_model = poptorch.inferenceModel(model)

    err_msg = (r"poptorch\.set_overlap_for_input applied on a node which is " +
               r"not a tensor input to the model\.")
    with pytest.raises(poptorch.poptorch_core.Error, match=err_msg):
        poptorch_model(torch.tensor([1.0]))

    class NormalModel(torch.nn.Module):
        def forward(self, x):
            x2 = poptorch.set_overlap_for_input(
                x, poptorch.OverlapMode.OverlapAccumulationLoop)
            y = x2 + 1
            return y

    model = NormalModel()
    poptorch_model = poptorch.inferenceModel(model)

    err_msg = (
        r"Overlapped IO is not supported with poptorch\.Pipelined" +
        r"Execution\. If you are using only one IPU, please switch to " +
        r"poptorch\.ShardedExecution\.")
    with pytest.raises(poptorch.poptorch_core.Error, match=err_msg):
        poptorch_model(torch.tensor([1.0]))

    opts = poptorch.Options()
    opts.setExecutionStrategy(poptorch.ShardedExecution())

    poptorch_model = poptorch.inferenceModel(model, options=opts)

    err_msg = (
        r"No IO tiles allocated\. You must allocate at least 32 IO tiles" +
        r" using poptorch\.Options\(\)\.TensorLocations\.numIOTiles\.")
    with pytest.raises(poptorch.poptorch_core.Error, match=err_msg):
        poptorch_model(torch.tensor([1.0]))

    opts.TensorLocations.numIOTiles(32)
    poptorch_model = poptorch.inferenceModel(model, options=opts)
    poptorch_model(torch.tensor([1.0]))
