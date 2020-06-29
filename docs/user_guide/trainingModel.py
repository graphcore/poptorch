# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch

# The native pytorch model.
model = torch.nn.Linear(10, 10)

# Some dummy inputs.
target = torch.randn(10)
input = torch.randn(10)

# Wrap the model in our PopTorch annotation wrapper.
poptorch_model = poptorch.trainingModel(model,
                                        device_iterations=1,
                                        replication_factor=1,
                                        gradient_accumulation=1,
                                        loss=torch.nn.MSELoss())

# Train on IPU.
for i in range(0, 2500):
    # Each call here executes the forward pass, loss calculation, and backward pass in one step.
    # Model input and loss function input are provided together.
    poptorch_out, loss = poptorch_model(input, target)

# Copy the trained weights from the IPU back into the host model.
poptorch_model.copyWeightsToHost()

# Execute the trained weights on host.
native_out = model(input)

# Models should be very close to native output although some operations are numerically
# different and floating point differences can accumulate.
assert torch.allclose(native_out, poptorch_out)
