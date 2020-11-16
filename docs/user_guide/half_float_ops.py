# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import poptorch
import torch

# pragma pylint: disable=function-redefined


# zero_res_start
## torch.ones and zeros
class Model(torch.nn.Module):
    def forward(self, x):
        # dtype is ignored, however the type is resolved to be the type of x
        return torch.zeros((2, 3, 4), dtype=torch.float32) + x


native_model = Model()

float16_tensor = torch.tensor([1.0], dtype=torch.float16)
float32_tensor = torch.tensor([1.0], dtype=torch.float32)

# The native model always yields a float32 tensor
assert native_model(float16_tensor).dtype == torch.float32
assert native_model(float32_tensor).dtype == torch.float32

# The poptorch model will resolve to the type of x
poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float16_tensor).dtype == torch.float16

poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float32_tensor).dtype == torch.float32

# zero_res_end


# rand_res_start
## torch.rand
class Model(torch.nn.Module):
    def forward(self, x):
        # dtype is ignored, however the type is resolved to be the type of x
        return torch.rand((2, 3, 4), dtype=torch.float32) + x


native_model = Model()

float16_tensor = torch.tensor([1.0], dtype=torch.float16)
float32_tensor = torch.tensor([1.0], dtype=torch.float32)

# The native model always yields a float32 tensor
assert native_model(float16_tensor).dtype == torch.float32
assert native_model(float32_tensor).dtype == torch.float32

# The poptorch model will resolve to the type of x
poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float16_tensor).dtype == torch.float16

poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float32_tensor).dtype == torch.float32

# rand_res_end


# uniform_res_start
## torch.distributions.uniform.Uniform
class Model(torch.nn.Module):
    def forward(self, x):
        # dtype is ignored, however the type is resolved to be the type of x
        ud = torch.distributions.uniform.Uniform(
            torch.tensor([10.0], dtype=torch.float16),
            torch.tensor([10.0], dtype=torch.float32))
        return ud.sample((10, 10, 1000)) + x


native_model = Model()

float16_tensor = torch.tensor([1.0], dtype=torch.float16)
float32_tensor = torch.tensor([1.0], dtype=torch.float32)

# The native model always yields a float32 tensor
assert native_model(float16_tensor).dtype == torch.float32
assert native_model(float32_tensor).dtype == torch.float32

# The poptorch model will resolve to the type of x
poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float16_tensor).dtype == torch.float16

poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float32_tensor).dtype == torch.float16
# uniform_res_end
