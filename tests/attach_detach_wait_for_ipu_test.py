#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import math
import os
import time
import unittest.mock
import pytest
import torch
import torch.multiprocessing as mp
import helpers
import poptorch


def inference_process(event, trace_model):
    assert os.environ.get('POPTORCH_WAIT_FOR_IPU') is not None

    torch.manual_seed(42)

    target = torch.randint(0, 10, [1])
    target = target.expand([10])

    model = torch.nn.Linear(10, 10)

    opts = poptorch.Options()
    # Ensure that both models use the same IPU
    opts.useIpuId(1)
    opts.Jit.traceModel(trace_model)

    inference = poptorch.inferenceModel(model, options=opts)
    inference.compile(torch.randn(10))
    event.set()
    time.sleep(12)
    inference.detachFromDevice()


@helpers.printCapfdOnExit
@unittest.mock.patch.dict("os.environ", {"POPTORCH_WAIT_FOR_IPU": "1"})
@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed to test POPTORCH_WAIT_FOR_IPU")
@helpers.overridePoptorchLogLevel("TRACE")
@pytest.mark.parametrize("trace_model", [True, False])
def test_attach_detach_wait_for_ipu(capfd, trace_model):

    torch.manual_seed(42)

    target = torch.randint(0, 10, [1])
    target = target.expand([10])
    input = torch.randn(10, 10)

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

    opts = poptorch.Options()
    # Ensure that both models use the same IPU
    opts.useIpuId(1)

    poptorch_model = poptorch.trainingModel(model, options=opts)

    ctx = mp.get_context('spawn')
    mgr = mp.Manager()
    event = mgr.Event()
    process = ctx.Process(target=inference_process, args=(event, trace_model))

    process.start()
    event.wait()
    _, initial_loss = poptorch_model(input, target)
    process.join()

    if math.isnan(initial_loss):
        raise ValueError("original_loss is NaN")

    poptorch_model.detachFromDevice()
    log = helpers.LogChecker(capfd)
    log.assert_contains("No IPU available, sleeping")
