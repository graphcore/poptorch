# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import poptorch
import torch


# print_tensor_start
class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        x += 1

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

for _ in range(1000):
    out, loss = poptorch_model(input=torch.tensor([1.0, 2.0, 3.0]),
                               target=torch.tensor([3.0, 4.0, 5.0]))
    print(f"Out = {out}, loss = {float(loss):.2f}")

print(f"Bias after training: {model_with_loss.model.bias}")

torch.testing.assert_allclose(model_with_loss.model.bias, 1.0)
