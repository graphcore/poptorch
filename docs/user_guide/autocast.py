# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch


# decorator_autocast_start
class MyModel(torch.nn.Module):
    @poptorch.autocast()
    def forward(self, x, y):
        return torch.bmm(x, y)


# decorator_autocast_end

# model_autocast_start
model = MyModel()
model.autocast()
poptorch_model = poptorch.inferenceModel(model)
# model_autocast_end

# layer_autocast_start
model = torch.nn.Sequential()
model.add_module('conv1', torch.nn.Conv2d(1, 20, 5))
model.add_module('relu1', torch.nn.ReLU())
model.add_module('conv2', torch.nn.Conv2d(20, 64, 5))
model.add_module('relu2', torch.nn.ReLU())
model.autocast()
model.relu1.autocast(False)
model.conv2.autocast(False)

# layer_autocast_end

# block_autocast_start
x = torch.randn(1, 10, 10)
y = torch.randn(1, 10, 10)
with poptorch.autocast():
    z = torch.bmm(x, y)
# block_autocast_end

# disable_autocast_start
opts = poptorch.Options()
opts.Precision.autocastEnabled(False)
poptorch_model = poptorch.inferenceModel(model, opts)
# disable_autocast_end

# policy_autocast_start
fp16 = [torch.nn.Conv2d, torch.relu]
fp32 = [torch.bmm]
promote = [torch.dot]
demote = []
policy = poptorch.autocasting.Policy(fp16, fp32, promote, demote)

opts = poptorch.Options()
opts.Precision.autocastPolicy(policy)
poptorch.model = poptorch.inferenceModel(model, opts)
# policy_autocast_end
