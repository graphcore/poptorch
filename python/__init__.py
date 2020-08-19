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


def trainingModel(model, options=None, optimizer=None):
    return _impl.PoplarExecutor(model=model,
                                options=options,
                                training=True,
                                optimizer=optimizer)


def inferenceModel(model, options=None):
    return _impl.PoplarExecutor(model=model, options=options, training=False)


def propagateInputShapes(graph, dummyInputs):
    for graphInput, dummyInput in zip(graph.inputs(), dummyInputs):
        graphInput.inferTypeFrom(dummyInput)
    poptorch_core.propagateInputShapes(graph)
