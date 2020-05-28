import torch
import torch.nn as nn
import poptorch

# This simple example demoonstrates compiling a model to add
# two tensors together using the IPU.


class SimpleAdder(nn.Module):
    def __init__(self):
        super(SimpleAdder, self).__init__()

    def forward(self, x, y):
        return x + y


model = SimpleAdder()
inference_model = poptorch.inferenceModel(model)

t1 = torch.tensor([1.])
t2 = torch.tensor([2.])

assert inference_model((t1, t2)) == 3.0
print("Success")
