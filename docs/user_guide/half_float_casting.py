# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
# pragma pylint: disable=function-redefined
import sys
import torch
import poptorch

if not poptorch.hasMLIRSupportOnPlatform():
    sys.exit(0)


# Cases in which casting resolves to the correct type
# correct_cast_start
class Model(torch.nn.Module):
    def forward(self, x, y):
        # In spite of "y.dtype" being ignored if it is float32, the dtype used
        # for the cast resolves to be the type of y because of the "+ y"
        return x.to(y.dtype) + y


native_model = Model()

float16_tensor = torch.tensor([1.0], dtype=torch.float16)
float32_tensor = torch.tensor([1.0], dtype=torch.float32)

assert native_model(float16_tensor, float16_tensor).dtype == torch.float16
assert native_model(float16_tensor, float32_tensor).dtype == torch.float32
assert native_model(float32_tensor, float16_tensor).dtype == torch.float16
assert native_model(float32_tensor, float32_tensor).dtype == torch.float32

poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float16_tensor, float16_tensor).dtype == torch.float16

poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float16_tensor, float32_tensor).dtype == torch.float32

poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float32_tensor, float16_tensor).dtype == torch.float16

poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float32_tensor, float32_tensor).dtype == torch.float32

# correct_cast_end


# Cases in which casting resolves to an incorrect type
# incorrect_cast_start
class Model(torch.nn.Module):
    def forward(self, x, y):
        # torch.float32 is ignored and the type is resolved to be the type of y
        return x.to(torch.float32) + y


native_model = Model()

float16_tensor = torch.tensor([1.0], dtype=torch.float16)
float32_tensor = torch.tensor([1.0], dtype=torch.float32)

assert native_model(float16_tensor, float16_tensor).dtype == torch.float32
assert native_model(float32_tensor, float16_tensor).dtype == torch.float32

opts = poptorch.Options()
# Important: this is only needed for traceModel(True)
opts.Jit.traceModel(True)
opts.Precision.halfFloatCasting(
    poptorch.HalfFloatCastingBehavior.HalfUpcastToFloat)

# This incorrectly results in a float 16 tensor
poptorch_model = poptorch.inferenceModel(native_model, opts)
assert poptorch_model(float16_tensor, float16_tensor).dtype == torch.float16

# This incorrectly results in a float 16 tensor
poptorch_model = poptorch.inferenceModel(native_model, opts)
assert poptorch_model(float32_tensor, float16_tensor).dtype == torch.float16

# UPDATE: with the new default of traceModel(False) PopTorch now matches the native behaviour
poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float16_tensor, float16_tensor).dtype == native_model(
    float16_tensor, float16_tensor).dtype

poptorch_model = poptorch.inferenceModel(native_model)
assert poptorch_model(float32_tensor, float16_tensor).dtype == native_model(
    float32_tensor, float16_tensor).dtype
# incorrect_cast_end
