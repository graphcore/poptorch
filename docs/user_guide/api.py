# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch


# print_tensor_start
class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        x = x + 1

        # It is important to make sure the result of the print is used.
        x = poptorch.ipu_print_tensor(x)

        return x + self.bias


# print_tensor_end

model = poptorch.inferenceModel(ExampleModel())
model(torch.tensor([1.0, 2.0, 3.0]))


# identity_start
def custom_loss(output, target):
    # Mean squared error with a scale
    loss = output - target
    loss = loss * loss * 5
    return poptorch.identity_loss(loss, reduction="mean")


class ExampleModelWithCustomLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ExampleModel()

    def forward(self, input, target):
        out = self.model(input)
        return out, custom_loss(out, target)


# identity_end

model_with_loss = ExampleModelWithCustomLoss()
poptorch_model = poptorch.trainingModel(model_with_loss)

print(f"Bias before training: {model_with_loss.model.bias}")

for _ in range(100):
    out, loss = poptorch_model(input=torch.tensor([1.0, 2.0, 3.0]),
                               target=torch.tensor([3.0, 4.0, 5.0]))
    print(f"Out = {out}, loss = {float(loss):.2f}")

print(f"Bias after training: {model_with_loss.model.bias}")

torch.testing.assert_allclose(model_with_loss.model.bias, 1.0)
poptorch_model.destroy()

model = ExampleModelWithCustomLoss()
input = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([3.0, 4.0, 5.0])
options = poptorch.Options()
# optim_start
opt = poptorch.optim.SGD(model.parameters(),
                         lr=0.01,
                         loss_scaling=2.0,
                         velocity_scaling=2.0)
poptorch_model = poptorch.trainingModel(model, options, opt)
poptorch_model(input, target)
# Update optimizer attribute
opt.loss_scaling = 1.0
# Update param_group attribute
opt.param_groups[0]["velocity_scaling"] = 1.0
# Set the new optimizer in the model
poptorch_model.setOptimizer(opt)
poptorch_model(input, target)
# optim_end
poptorch_model.destroy()

# optim_const_start
# lr, momentum and velocity_scaling will be marked as variable.
opt = poptorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
# momentum and velocity_scaling  will be marked as constant.
opt = poptorch.optim.SGD(model.parameters(), lr=0.01)
# lr and momentum will be marked as variable.
# velocity_scaling will be marked as constant.
opt = poptorch.optim.SGD(model.parameters(),
                         lr=0.01,
                         momentum=0.0,
                         velocity_scaling=2.0)
opt.variable_attrs.markAsConstant("velocity_scaling")
# lr, momentum and velocity_scaling will be marked as variable.
opt = poptorch.optim.SGD(model.parameters(), lr=0.01, velocity_scaling=2.0)
opt.variable_attrs.markAsVariable("momentum")
# optim_const_end

# torch_optim_const_start
# momentum will be marked as constant (It's not set)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
# lr will be marked as variable.
# momentum will still be marked as constant (Because its default value is 0.0)
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
# lr and momentum will both be marked as variable.
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=1.0)
# torch_optim_const_end
