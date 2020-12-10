# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import poptorch
# If running on the model then make sure to run on the full size model to
# avoid running out of memory.
if not poptorch.ipuHardwareIsAvailable():
    os.environ["POPTORCH_IPU_MODEL"] = "1"

# pylint: disable=reimported
# pylint: disable=ungrouped-imports
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position

# inference_model_start
import torch
import torchvision
import poptorch

# Some dummy imagenet sized input.
picture_of_a_cat_here = torch.randn([1, 3, 224, 224])

# The model, in this case a MobileNet model with pretrained weights that comes
# canned with Pytorch.
model = torchvision.models.mobilenet_v2(pretrained=True)
model.train(False)

# Wrap in the PopTorch inference wrapper
inference_model = poptorch.inferenceModel(model)

# Execute on IPU.
out_tensor = inference_model(picture_of_a_cat_here)

# Get the top 5 ImageNet classes.
top_five_classes = torch.topk(torch.softmax(out_tensor, 1), 5)
print(top_five_classes)

# Try the same on native PyTorch
native_out = model(picture_of_a_cat_here)

native_top_five_classes = torch.topk(torch.softmax(native_out, 1), 5)

# Models should be very close to native output although some operations are
# numerically different and floating point differences can accumulate.
assert any(top_five_classes[1][0] == native_top_five_classes[1][0])
# inference_half_start
model = torch.nn.Linear(1, 10).half()
t1 = torch.tensor([1.]).half()

inference_model = poptorch.inferenceModel(model)
out = inference_model(t1)

assert out.dtype == torch.half
# inference_half_end
