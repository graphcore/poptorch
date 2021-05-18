#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
import poptorch


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
opts.anchorTensor('grad_bias', 'Gradient___model.fc2.bias')
opts.anchorTensor('update_weight', 'UpdatedVar___model.fc2.weight')
# tensor_anchor_end

# tensor_retrieve_start
poptorch_model = poptorch.trainingModel(model, opts)
poptorch_model(input, label)

grad = poptorch_model.getAnchoredTensor('grad_bias')
update = poptorch_model.getAnchoredTensor('update_weight')
# tensor_retrieve_end