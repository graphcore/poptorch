import torch
import torch.nn as nn
import numpy as np
import poptorch


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(self, x, y):

        t1 = (x + y)
        t2 = (t1, x * y)

        return t2, y - x, t2[1] + t1


# Create our model.
model = Network()

inference_model = poptorch.inferenceModel(model)

x = torch.ones(2)
y = torch.zeros(2)

print(inference_model((x, y)))
print(model(x, y))
