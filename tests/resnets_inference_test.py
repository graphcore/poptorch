#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torchvision.models as models
import poptorch

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

# Models here are hopefuly representative of their cousins (i.e test Resnet18 without testing Resnet-34/50/101/152)
# The others will be tested in hardware benchmark tests,
tested_subset = [
    models.resnet18,
    models.resnext50_32x4d,
    models.mnasnet1_0,
    models.mobilenet_v2,
]

# Deliberately un-tested models
unsupported_models = [
    models.vgg11,  # Supported but takes a long time to compile.
    models.shufflenet_v2_x1_0,  # Supported but takes a long time to compile.
    models.densenet121,  # Supported but takes a long time to compile.
    models.wide_resnet50_2,  # Supported but doesn't fit on 1 IPU.
    # Supported on IPU_MODEL but runs into stream limit on IPU.
    models.alexnet,
    models.inception_v3,  # TODO(T26199): Output mismatch
]


def inference_harness(imagenet_model):
    torch.manual_seed(42)

    image_input = torch.randn([1, 3, 224, 224])

    # We are running on a dummy input so it doesn't matter if the weights are trained.
    model = imagenet_model(pretrained=False)
    model.eval()

    # Run on CPU.
    nativeOut = model(image_input)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(image_input)

    torch.testing.assert_allclose(nativeOut,
                                  poptorch_out,
                                  atol=1e-05,
                                  rtol=0.1)

    native_class = torch.topk(torch.softmax(nativeOut, 1), 5)
    pop_class = torch.topk(torch.softmax(poptorch_out, 1), 5)

    assert torch.equal(native_class.indices, pop_class.indices)
    torch.testing.assert_allclose(native_class.values, pop_class.values)


def test_resnet18():
    inference_harness(models.resnet18)


def test_resnext50_32x4d():
    inference_harness(models.resnext50_32x4d)


def test_mnasnet1_0():
    inference_harness(models.mnasnet1_0)


def test_mobilenet_v2():
    inference_harness(models.mobilenet_v2)


def test_googlenet():
    inference_harness(models.googlenet)
