#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import readline
import rlcompleter
readline.parse_and_bind('tab: complete')

import torch
import torch.nn as nn
import numpy as np
import os

import poptorch
import torchvision.models as models

from PIL import Image
from torchvision import transforms

# Image loading from https://pytorch.org/hub/pytorch_vision_resnet/
this_dir = os.path.dirname(os.path.realpath(__file__))
input_image = Image.open(os.path.join(this_dir, "zeus.jpg"))
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                          0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

supported_models = [
    models.resnet18,
    models.resnet50,
    models.alexnet,
    models.vgg16,
    models.resnext50_32x4d,
]

for model in supported_models:
    print("Loading model: " + str(model))
    model = model(pretrained=True)

    model.eval()

    inference_model = poptorch.inferenceModel(model)
    out_tensor = inference_model(input_batch)

    print(torch.topk(torch.softmax(out_tensor, 1), 5))
    print(torch.topk(torch.softmax(model(input_batch), 1), 5))
