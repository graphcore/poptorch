# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

x = self.layer1(x)

# It is important to make sure the result of the print is used.
x = poptorch.ipu_print_tensor(x)

x = self.layer2(x)


class CustomLoss(torch.nn.Module):
    def forward(self, x, target):
        # Mean squared error with a scale
        loss = x - target
        loss = loss * loss * 5
        return poptorch.identity_loss(loss, reduction="mean")


poptorch_model = poptorch.trainingModel(model,
                                        device_iterations=1,
                                        loss=CustomLoss())
