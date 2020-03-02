from poptorch_core import *
import poptorch_core
import torch
import os
import sys


import torch
import torch.nn as nn

# Don't think we should keep this. It's done in popart, but we should just be able to add this path to LD_LIBRARY_PATH in build/activate.sh?
lp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
lp = os.path.abspath(lp)
sys.path.insert(0, lp)


pipeline_stage = torch.ops.poptorch.pipeline_stage
virtual_graph = torch.ops.poptorch.virtual_graph
ipu_print_tensor = torch.ops.poptorch.ipu_print_tensor

# From pytorch/torch/jit/__init__.py
def make_tuple(example_inputs):
    if isinstance(example_inputs, (torch.Tensor, dict)):
        return (example_inputs,)
    # done primarily so that weird iterables fail here and not pybind11 code
    if not isinstance(example_inputs, tuple):
        return tuple(example_inputs)
    return example_inputs


class PoplarExecutor:
    def __init__(self, model, training, device_iterations):
        self.executable = None
        self.model = model
        self.training = training
        self.device_iterations = device_iterations

    def __call__(self, tensors):

        # Convert single tensor to tuple.
        in_tensors = make_tuple(tensors)

        if self.executable == None:
            print("First time call to model will invoke poplar compilation. " +
                  str(self.device_iterations) + " " + str(self.training))

            # Input will be in form of [BatchSize* BatchPerStep, ...] so we should slice it up so we compile by the batch size alone.
            newTuple = []
            for tensor in in_tensors:
                newTuple.append(tensor.narrow(
                    0, 0, tensor.size()[0] // self.device_iterations))

            # Compile the poplar executable based on the batchsize.
            newTuple = tuple(newTuple)
            n = torch.jit.trace(self.model, newTuple)
            self.executable = poptorch_core.compile(
                n._c, newTuple, self.device_iterations, self.training)

        # Execute the poplar executable with the full size (batch * device interations)
        return poptorch_core.execute(self.executable, in_tensors)


def trainingModel(model, device_iterations):

    class ModelTrainingWrapper(nn.Module):
        def __init__(self, model):
            super(ModelTrainingWrapper, self).__init__()
            self.model = model

        def __call__(self, args, labels):
            return self.model(args)

    wrappedModel = ModelTrainingWrapper(model)
    return PoplarExecutor(wrappedModel, True, device_iterations)


def inferenceModel(model):
    return PoplarExecutor(model, False, 1)
