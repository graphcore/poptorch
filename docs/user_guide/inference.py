# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import torchvision.models as models

# Some dummy imagenet sized input.
picture_of_a_cat_here = torch.randn([1, 3, 224, 224])

# Our model, in this case a ResNext model with pretrained weights that comes
# canned with Pytorch.
model = models.resnext101_32x2d(pretrained=True)

# Wrap in the PopTorch inference wrapper
inference_model = poptorch.inferenceModel(model)

# Execute on IPU.
out_tensor = inference_model(picture_of_a_cat_here)

# Get the top 5 ImageNet classes.
top_five_classes = torch.topk(torch.softmax(out_tensor, 1), 5))