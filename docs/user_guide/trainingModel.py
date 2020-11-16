# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
# training_model_start
import poptorch
import torch


class ExampleModelWithLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)
        self.loss = torch.nn.MSELoss()

    def forward(self, x, target=None):
        fc = self.fc(x)
        if self.training:
            return fc, self.loss(fc, target)
        return fc


model = ExampleModelWithLoss()

# Wrap the model in our PopTorch annotation wrapper.
poptorch_model = poptorch.trainingModel(model)

# Some dummy inputs.
input = torch.randn(10)
target = torch.randn(10)

# Train on IPU.
for i in range(0, 100):
    # Each call here executes the forward pass, loss calculation, and backward
    # pass in one step.
    # Model input and loss function input are provided together.
    poptorch_out, loss = poptorch_model(input, target)
    print(f"{i}: {loss}")

# Copy the trained weights from the IPU back into the host model.
poptorch_model.copyWeightsToHost()

# Execute the trained weights on host.
model.eval()
native_out = model(input)

# Models should be very close to native output although some operations are
# numerically different and floating point differences can accumulate.
assert torch.allclose(native_out, poptorch_out)
