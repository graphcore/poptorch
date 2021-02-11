#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import math
import os
import time
import unittest.mock
import pytest
import helpers
import torch
import torch.multiprocessing as mp
import poptorch


def inference_process(event):
    assert os.environ.get('POPTORCH_WAIT_FOR_IPU') is not None

    torch.manual_seed(42)

    target = torch.randint(0, 10, [1])
    target = target.expand([10])

    model = torch.nn.Linear(10, 10)

    opts = poptorch.Options()
    # Ensure that both models use the same IPU
    opts.useIpuId(1)

    inference = poptorch.inferenceModel(model, options=opts)
    inference.compile(torch.randn(10))
    event.set()
    time.sleep(12)
    inference.detachFromDevice()


@helpers.printCapfdOnExit
@unittest.mock.patch.dict("os.environ", {"POPTORCH_WAIT_FOR_IPU": "1"})
@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed to test POPTORCH_WAIT_FOR_IPU")
def test_attach_detach_wait_for_ipu(capfd):
    poptorch.setLogLevel('TRACE')  # Force trace logging

    torch.manual_seed(42)

    target = torch.randint(0, 10, [1])
    target = target.expand([10])
    input = torch.randn(10, 10)

    model = torch.nn.Linear(10, 10)

    opts = poptorch.Options()
    # Ensure that both models use the same IPU
    opts.useIpuId(1)

    training = helpers.trainingModelWithLoss(model,
                                             options=opts,
                                             loss=torch.nn.CrossEntropyLoss())

    ctx = mp.get_context('spawn')
    mgr = mp.Manager()
    event = mgr.Event()
    process = ctx.Process(target=inference_process, args=(event, ))

    process.start()
    event.wait()
    _, initial_loss = training(input, target)
    process.join()

    if math.isnan(initial_loss):
        raise ValueError("original_loss is NaN")

    training.detachFromDevice()
    log = helpers.LogChecker(capfd)
    log.assert_contains("No IPU available, sleeping")
