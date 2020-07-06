#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import numpy as np
import poptorch
import poptorch.testing


def test_jit_script():
    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    # Create our model.
    model = Network()
    opts = poptorch.Options()
    opts.Jit.traceModel(False)
    inference_model = poptorch.inferenceModel(model, opts)

    x = torch.ones(2)
    y = torch.zeros(2)

    ipu = inference_model(x, y)
    ref = model(x, y)
    assert poptorch.testing.allclose(
        ref, ipu), "%s doesn't match the expected output %s" % (ipu, ref)


def test_set_options():
    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    # Create our model.
    model = Network()
    opts = poptorch.Options()
    # Just set a bunch of options and check they're successfully parsed.
    opts.deviceIterations(1).enablePipelining(False).replicationFactor(
        1).logDir("/tmp").profile(False)
    inference_model = poptorch.inferenceModel(model, opts)

    x = torch.ones(2)
    y = torch.zeros(2)

    ipu = inference_model(x, y)


def test_set_popart_options():
    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    # Create our model.
    model = Network()
    opts = poptorch.Options()
    opts.Popart.Set("hardwareInstrumentations", set([0, 1]))
    opts.Popart.Set("dotChecks", [0, 1])
    opts.Popart.Set("engineOptions", {
        "debug.allowOutOfMemory": "true",
        "exchange.streamBufferOverlap": "any"
    })
    opts.Popart.Set("customCodelets", [])
    opts.Popart.Set("autoRecomputation", 1)
    opts.Popart.Set("cachePath", "/tmp")
    opts.Popart.Set("enableOutlining", True)
    inference_model = poptorch.inferenceModel(model, opts)
    x = torch.ones(2)
    y = torch.zeros(2)

    ipu = inference_model(x, y)
