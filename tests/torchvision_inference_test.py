#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os  # pylint: disable=unused-import
import unittest.mock
import pytest
import torch
import torchvision.models as models
import poptorch
import helpers

# Torchvision models.
# AlexNet
# VGG-11
# VGG-13
# VGG-16
# VGG-19
# VGG-11 with batch normalization
# VGG-13 with batch normalization
# VGG-16 with batch normalization
# VGG-19 with batch normalization
# ResNet-18
# ResNet-34
# ResNet-50
# ResNet-101
# ResNet-152
# SqueezeNet 1.0
# SqueezeNet 1.1
# Densenet-121
# Densenet-169
# Densenet-201
# Densenet-161
# Inception v3
# GoogleNet
# ShuffleNet V2
# MobileNet V2
# ResNeXt-50-32x4d
# ResNeXt-101-32x8d
# Wide ResNet-50-2
# Wide ResNet-101-2
# MNASNet 1.0

# Models here are hopefully representative of their cousins (i.e test Resnet18 without testing Resnet-34/50/101/152)
# The others will be tested in hardware benchmark tests,
tested_models = [
    models.resnet18,
    models.resnext50_32x4d,
    models.mnasnet1_0,
    models.mobilenet_v2,
    models.googlenet,
    models.inception_v3,
    # SqueezeNet v1.0 simply has more parameters and a greater computational cost
    models.squeezenet1_1,
]

# Deliberately un-tested models
untested_models = [
    models.vgg11,  # Supported but takes a long time to compile.
    models.shufflenet_v2_x1_0,  # Supported but takes a long time to compile.
    models.densenet121,  # Supported but takes a long time to compile.
    models.wide_resnet50_2,  # Supported but doesn't fit on 1 IPU.
    # Supported on IPU_MODEL but runs into stream limit on IPU.
    models.alexnet,
]


def inference_harness(imagenet_model, trace_model):
    torch.manual_seed(42)

    image_input = torch.randn([1, 3, 224, 224])

    # We are running on a dummy input so it doesn't matter if the weights are trained.
    model = imagenet_model(pretrained=False)
    model.eval()

    # Run on CPU.
    native_out = model(image_input)

    opts = poptorch.Options()
    opts.Jit.traceModel(trace_model)

    poptorch_model = poptorch.inferenceModel(model, opts)

    poptorch_out = poptorch_model(image_input)

    helpers.assert_allclose(expected=native_out,
                            actual=poptorch_out,
                            atol=1e-05,
                            rtol=0.1)

    native_class = torch.topk(torch.softmax(native_out, 1), 5)
    pop_class = torch.topk(torch.softmax(poptorch_out, 1), 5)

    helpers.assert_allequal(expected=native_class.indices,
                            actual=pop_class.indices)
    helpers.assert_allclose(expected=native_class.values,
                            actual=pop_class.values)


@unittest.mock.patch.dict("os.environ", helpers.disableSmallModel())
@pytest.mark.parametrize("model", tested_models + untested_models)
@pytest.mark.parametrize("trace_model", [True, False])
@pytest.mark.extendedTestingOnly
def test_model(model, trace_model):
    if model in untested_models:
        pytest.skip("Model not currently tested")
    inference_harness(model, trace_model)
