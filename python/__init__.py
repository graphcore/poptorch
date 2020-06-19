# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from poptorch.poptorch_core import *
import poptorch.poptorch_core as poptorch_core

begin_ipu_block = torch.ops.poptorch.begin_ipu_block
end_ipu_block = torch.ops.poptorch.end_ipu_block
ipu_print_tensor = torch.ops.poptorch.ipu_print_tensor


# From pytorch/torch/jit/__init__.py
def make_tuple(example_inputs):
    if isinstance(example_inputs, (torch.Tensor, dict)):
        return (example_inputs, )
    # done primarily so that weird iterables fail here and not pybind11 code
    if not isinstance(example_inputs, tuple):
        return tuple(example_inputs)
    return example_inputs


def identity_loss(x, reduction="none"):
    if (reduction == "sum"):
        return torch.ops.poptorch.identity_loss(x, 0)

    if (reduction == "mean"):
        return torch.ops.poptorch.identity_loss(x, 1)

    assert reduction == "none", "Unsupported reduction type!"
    return torch.ops.poptorch.identity_loss(x, 2)


def convertOptimizerToDict(optimizer):
    if len(optimizer.param_groups) != 1:
        print(
            "Error: Poptorch currently only supports one parameter group! (all parameters)"
        )
        exit()

    return {
        "lr": (optimizer.param_groups[0]["lr"], False),
        "momentum": (optimizer.param_groups[0]["momentum"], False),
        "weight_decay": (optimizer.param_groups[0]["weight_decay"], False),
        "dampening": (optimizer.param_groups[0]["dampening"], False)
    }


identity_loss = identity_loss


class IPU(nn.Module):
    def __init__(self, ipu_id, layer_to_call=None):
        super(IPU, self).__init__()

        self.ipu_id = ipu_id
        self.layer_to_call = layer_to_call

    def __enter__(self):
        begin_ipu_block(self.ipu_id)

    def __exit__(self, type, value, traceback):
        end_ipu_block()

    def __call__(self, x):
        begin_ipu_block(self.ipu_id)
        out = self.layer_to_call(x)
        return out


class PoplarExecutor:
    def __init__(self,
                 model,
                 training,
                 device_iterations,
                 trace_model,
                 replication_factor=1,
                 gradient_accumulation=1,
                 profile=False,
                 optimizer={}):
        self.executable = None
        self.model = model
        self.training = training
        self.device_iterations = device_iterations
        self.gradient_accumulation = gradient_accumulation
        self.replication_factor = replication_factor
        self.profile = profile
        self.trace_model = trace_model
        self.optimizer = optimizer

    def __call__(self, tensors, optimizer=None):
        # Convert single tensor to tuple.
        in_tensors = make_tuple(tensors)

        if self.executable == None:
            print("First time call to model will invoke poplar compilation. " +
                  str(self.device_iterations) + " " + str(self.training))

            # Input will be in form of [BatchSize* BatchPerStep, ...] so we should slice it up so we compile by the batch size alone.
            newTuple = []

            extra_poplar_batch_dims = self.device_iterations * self.replication_factor * self.gradient_accumulation

            for tensor in in_tensors:
                newTuple.append(
                    tensor.narrow(0, 0,
                                  tensor.size()[0] // extra_poplar_batch_dims))

            # Compile the poplar executable based on the batchsize.
            newTuple = tuple(newTuple)

            if self.trace_model:
                print('Compiling the model using tracing')
                n = torch.jit.trace(self.model, newTuple)

                self.executable = compileWithTrace(
                    n._c, newTuple, self.device_iterations, self.training,
                    self.replication_factor, self.gradient_accumulation,
                    self.optimizer, self.profile)
            else:
                print('Compiling the model using scripting')
                n = torch.jit.script(self.model)
                graphInputs = list(n.graph.inputs())
                for graphInput, argIn in zip(graphInputs[1:], newTuple):
                    if isinstance(argIn, torch.Tensor):
                        graphInput.inferTypeFrom(argIn)

                self.executable = compileWithScript(
                    n._c, n.graph, newTuple, self.device_iterations,
                    self.training, self.replication_factor,
                    self.gradient_accumulation, self.profile)

        # Execute the poplar executable with the full size (batch * device interations)
        if optimizer:
            self.optimizer = optimizer
            output = execute(self.executable, in_tensors,
                             convertOptimizerToDict(self.optimizer))
        else:
            output = execute(self.executable, in_tensors, {})

        if len(output) > 1:
            return tuple(output)
        else:
            return output[0]


def trainingModel(model,
                  device_iterations,
                  gradient_accumulation=1,
                  profile=False,
                  trace_model=True,
                  loss=None,
                  optimizer=None):
    if optimizer == None:
        optimizer = optim.SGD(model.parameters(), lr=0.01)

    optimizer = convertOptimizerToDict(optimizer)

    class ModelTrainingWrapper(nn.Module):
        def __init__(self, model, loss=None):
            super(ModelTrainingWrapper, self).__init__()
            self.model = model
            self.loss = loss

        def __call__(self, args, loss_inputs):
            output = self.model(args)

            if self.loss:
                loss = self.loss(output, loss_inputs)
                return output, loss

            return output

    wrappedModel = ModelTrainingWrapper(model, loss)
    return PoplarExecutor(wrappedModel,
                          True,
                          device_iterations,
                          gradient_accumulation=gradient_accumulation,
                          profile=profile,
                          trace_model=trace_model,
                          optimizer=optimizer)


def inferenceModel(model, device_iterations=1, profile=False,
                   trace_model=True):
    return PoplarExecutor(model,
                          False,
                          device_iterations,
                          profile=profile,
                          trace_model=trace_model)


def propagateInputShapes(graph, dummyInputs):
    for graphInput, dummyInput in zip(graph.inputs(), dummyInputs):
        graphInput.inferTypeFrom(dummyInput)
    poptorch_core.propagateInputShapes(graph)
