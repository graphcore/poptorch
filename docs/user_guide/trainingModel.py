# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
# training_model_start
import torch
import poptorch


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


torch.manual_seed(0)
model = ExampleModelWithLoss()

# Wrap the model in our PopTorch annotation wrapper.
poptorch_model = poptorch.trainingModel(model)

# Some dummy inputs.
input = torch.randn(10)
target = torch.randn(10)

# Train on IPU.
for i in range(0, 800):
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
torch.testing.assert_allclose(native_out, poptorch_out, rtol=1e-04, atol=1e-04)
# training_model_end
Model = ExampleModelWithLoss


def train(_):
    pass


def validate(_):
    pass


# explicit_copy_start
model = Model()
poptorch_train = poptorch.trainingModel(model)
poptorch_inf = poptorch.inferenceModel(model)

train(poptorch_train)
torch.save(model.state_dict(), "model.save")  # OK
validate(poptorch_inf)  # OK
validate(model)  # OK

train(model)
# Explicit copy needed
poptorch_inf.copyWeightsToDevice()
validate(poptorch_inf)
# explicit_copy_end
