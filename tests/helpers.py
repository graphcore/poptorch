# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import poptorch


def trainingModelWithLoss(model, loss, options=None, optimizer=None):
    class TrainingModelWithLoss(torch.nn.Module):
        def __init__(self, model, loss):
            super().__init__()
            self._real_call = model.__call__
            # The original model *must* be stored in the wrapper
            # even if it's not used (The tracer will inspect it
            # for parameters).
            self._model = model
            self._loss = loss

        def forward(self, args, loss_inputs):  # pylint: disable=unused-argument
            assert False, ("Shouldn't be called, signature should match"
                           " the one of __call__")

        def __call__(self, args, loss_inputs):
            output = self._real_call(args)
            loss = self._loss(output, loss_inputs)
            return output, loss

    # Store the real __call__ method before PoplarExecutor wraps it
    return poptorch._impl.PoplarExecutor(  # pylint: disable=protected-access
        model=TrainingModelWithLoss(model, loss),
        options=options,
        training=True,
        optimizer=optimizer,
        user_model=model)
