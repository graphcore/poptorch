# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import torch.optim as optim

import poptorch.poptorch_core as poptorch_core
from poptorch.poptorch_core import ipuHardwareIsAvailable

from .logging import logger
from . import _impl
from .enums import *
from .ops import *
from .options import *


class IPU(nn.Module):
    def __init__(self, ipu_id, layer_to_call=None):
        super().__init__()

        self.ipu_id = ipu_id
        self.layer_to_call = layer_to_call

    def __enter__(self):
        begin_ipu_block(self.ipu_id)

    def __exit__(self, type, value, traceback):
        end_ipu_block()

    def __call__(self, *input, **kwargs):
        begin_ipu_block(self.ipu_id)
        out = self.layer_to_call(*input, **kwargs)
        return out


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 options,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 num_workers=0,
                 **kwargs):
        assert isinstance(options, Options)
        self._combined_batch_size = batch_size * \
            options.device_iterations * \
            options.replication_factor * \
            options.Training.gradient_accumulation

        super().__init__(dataset,
                         batch_size=self._combined_batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         **kwargs)

    @property
    def combinedBatchSize(self):
        return self._combined_batch_size


def trainingModel(model, options=None, loss=None, optimizer=None):
    options = options or Options()
    if options.defaultAnchorMode():
        # In training it makes sense to see only the last result, by default.
        options.anchorMode(AnchorMode.Final)
    if not optimizer:
        optimizer = optim.SGD(model.parameters(), lr=0.01)

    optimizer = _impl.convertOptimizerToDict(optimizer)

    class ModelTrainingWrapper(nn.Module):
        def __init__(self, model, loss=None):
            super().__init__()
            self.model = model
            self.loss = loss
            # Store the real __call__ method before PoplarExecutor wraps it
            self.real_model_call = model.__call__

        def __call__(self, args, loss_inputs):
            output = self.real_model_call(args)

            if self.loss:
                loss = self.loss(output, loss_inputs)
                return output, loss

            return output

    wrappedModel = ModelTrainingWrapper(model, loss)
    return _impl.PoplarExecutor(model=wrappedModel,
                                options=options,
                                training=True,
                                optimizer=optimizer)


def inferenceModel(model, options=None):
    options = options or Options()
    if options.defaultAnchorMode():
        # In inference it makes sense to see all the results, by default.
        options.anchorMode(AnchorMode.All)
    assert options.Training.gradient_accumulation == 1, (
        "Gradient accumulation"
        " should be left to its default value (1) for inference")

    return _impl.PoplarExecutor(model=model, options=options, training=False)


def propagateInputShapes(graph, dummyInputs):
    for graphInput, dummyInput in zip(graph.inputs(), dummyInputs):
        graphInput.inferTypeFrom(dummyInput)
    poptorch_core.propagateInputShapes(graph)
