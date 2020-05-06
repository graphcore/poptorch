import os
import sys
# Don't think we should keep this. It's done in popart, but we should just be able to add this path to LD_LIBRARY_PATH in build/activate.sh?
lp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
lp = os.path.abspath(lp)
sys.path.insert(0, lp)

import torch
import torch.nn as nn

from poptorch_core import *
import poptorch_core


begin_ipu_block = torch.ops.poptorch.begin_ipu_block
end_ipu_block = torch.ops.poptorch.end_ipu_block
ipu_print_tensor = torch.ops.poptorch.ipu_print_tensor

# From pytorch/torch/jit/__init__.py
def make_tuple(example_inputs):
    if isinstance(example_inputs, (torch.Tensor, dict)):
        return (example_inputs,)
    # done primarily so that weird iterables fail here and not pybind11 code
    if not isinstance(example_inputs, tuple):
        return tuple(example_inputs)
    return example_inputs


class IPU(nn.Module):
    def __init__(self, ipu_id, layer_to_call=None):
        super(IPU, self).__init__()

        self.ipu_id = ipu_id
        self.layer_to_call = layer_to_call

    def __enter__(self):
        begin_ipu_block(self.ipu_id)

    def __exit__(self ,type, value, traceback):
        end_ipu_block()

    def __call__(self, x):
        begin_ipu_block(self.ipu_id)
        out = self.layer_to_call(x)
        return out

class PoplarExecutor:
    def __init__(self, model, training, device_iterations, replication_factor=1, gradient_accumulation=1, profile=False):
        self.executable = None
        self.model = model
        self.training = training
        self.device_iterations = device_iterations
        self.gradient_accumulation = gradient_accumulation
        self.replication_factor = replication_factor
        self.profile = profile

    def __call__(self, tensors):

        # Convert single tensor to tuple.
        in_tensors = make_tuple(tensors)

        if self.executable == None:
            print("First time call to model will invoke poplar compilation. " +
                  str(self.device_iterations) + " " + str(self.training))

            # Input will be in form of [BatchSize* BatchPerStep, ...] so we should slice it up so we compile by the batch size alone.
            newTuple = []

            extra_poplar_batch_dims = self.device_iterations * self.replication_factor * self.gradient_accumulation

            for tensor in in_tensors:
                newTuple.append(tensor.narrow(
                    0, 0, tensor.size()[0] // extra_poplar_batch_dims))


            # Compile the poplar executable based on the batchsize.
            newTuple = tuple(newTuple)
            n = torch.jit.trace(self.model, newTuple)

            self.executable = poptorch_core.compile(
                n._c, newTuple, self.device_iterations, self.training, self.replication_factor, self.gradient_accumulation, self.profile)

        # Execute the poplar executable with the full size (batch * device interations)
        output = poptorch_core.execute(self.executable, in_tensors)

        if len(output) > 1:
            return tuple(output)
        else:
            return output[0]

def trainingModel(model, device_iterations, gradient_accumulation=1, profile=False):

    class ModelTrainingWrapper(nn.Module):
        def __init__(self, model):
            super(ModelTrainingWrapper, self).__init__()
            self.model = model

        def __call__(self, args, labels):
            return self.model(args)

    wrappedModel = ModelTrainingWrapper(model)
    return PoplarExecutor(wrappedModel, True, device_iterations, gradient_accumulation=gradient_accumulation, profile=profile)


def inferenceModel(model, device_iterations=1, profile=False):
    return PoplarExecutor(model, False, device_iterations, profile=profile)

