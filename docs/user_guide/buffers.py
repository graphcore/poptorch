# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
import poptorch


# counter_model_wrong_start
class CounterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.i = torch.tensor([0.], dtype=torch.float)

    def forward(self):
        self.i += 1
        return self.i


model = CounterModel()
poptorch_model = poptorch.inferenceModel(model)
print(poptorch_model())  # tensor([6.])
print(poptorch_model())  # tensor([6.])
# counter_model_wrong_end

torch.testing.assert_allclose(model.i, torch.tensor([5.], dtype=torch.float))


# pragma pylint: disable=function-redefined,no-member
# counter_model_correct_start
class CounterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("i", torch.tensor([0.], dtype=torch.float))

    def forward(self):
        self.i += 1
        return self.i


model = CounterModel()
poptorch_model = poptorch.inferenceModel(model)

print(poptorch_model())  # tensor([1.])
print(poptorch_model())  # tensor([2.])
# counter_model_correct_end

# Because the model is running in inference mode, we will need to manually
# call copyWeightsToHost
poptorch_model.copyWeightsToHost()
torch.testing.assert_allclose(model.i, torch.tensor([2.], dtype=torch.float))
