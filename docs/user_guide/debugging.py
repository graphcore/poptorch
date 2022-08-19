#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import sys
import torch
import poptorch

if not poptorch.hasMLIRSupportOnPlatform():
    sys.exit(0)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 10)
        self.loss = torch.nn.MSELoss(reduction="mean")

    def forward(self, x, labels=None):
        out = self.fc2(self.relu(self.fc1(x)))
        if self.training:
            return self.loss(out, labels)
        return out


# tensor_names_start
input = torch.rand(10, 10)
label = torch.rand(10, 10)

model = Model()
poptorch_model = poptorch.trainingModel(model)
poptorch_model(input, label)

tensor_names = poptorch_model.getTensorNames()
# tensor_names_end

# tensor_anchor_start
opts = poptorch.Options()
opts.anchorTensor('grad_bias', 'Gradient___fc2.bias')
opts.anchorTensor('update_weight', 'UpdatedVar___fc2.weight')
# tensor_anchor_end

poptorch_model.destroy()

# tensor_retrieve_start
poptorch_model = poptorch.trainingModel(model, opts)
poptorch_model(input, label)

grad = poptorch_model.getAnchoredTensor('grad_bias')
update = poptorch_model.getAnchoredTensor('update_weight')
# tensor_retrieve_end

poptorch_model.destroy()

# optim_state_dict_start
optim = poptorch.optim.SGD(model.parameters(), lr=0.01)
poptorch_model = poptorch.trainingModel(model, opts, optim)
poptorch_model(input, label)

state = optim.state_dict()
# optim_state_dict_end
