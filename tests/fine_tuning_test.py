#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import copy
import os  # pylint: disable=unused-import
import unittest.mock
import pytest
import torch
import torchvision.models as models
import helpers
import poptorch


def fine_tuning_harness(imagenet_model, trace_model):
    torch.manual_seed(42)

    num_classes = 2
    num_examples = 2
    num_epochs = 20

    data = torch.randn((num_examples, 3, 224, 224))
    target = torch.randint(0, num_classes, (num_examples, ))

    base_model = imagenet_model(pretrained=False)

    loss_fn = torch.nn.CrossEntropyLoss()

    class ModelWithLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = base_model

        def forward(self, data, target):
            out = base_model(data)
            loss = loss_fn(out, target)
            return out, loss

    model = ModelWithLoss()

    for param in model.base_model.parameters():
        param.requires_grad = False

    # Change the linear classifier at the top.
    model.base_model.fc = torch.nn.Linear(model.base_model.fc.in_features,
                                          num_classes)
    for param in model.base_model.fc.parameters():
        assert param.requires_grad
    initial_params = copy.deepcopy(model).state_dict()

    # Fine tune.
    optim = torch.optim.SGD(model.base_model.fc.parameters(), lr=0.001)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model,
                                            optimizer=optim,
                                            options=options)

    for _ in range(num_epochs):
        _ = poptorch_model(data, target)

    # Assert only the last layer was changed.
    for name, param in model.named_parameters():
        if name.startswith('base_model.fc'):
            assert not torch.allclose(param.data, initial_params[name])
        else:
            helpers.assert_allclose(actual=param.data,
                                    expected=initial_params[name])


@unittest.mock.patch.dict("os.environ", helpers.disableSmallModel())
@pytest.mark.parametrize("trace_model", [True, False])
def test_resnet18(trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): NotImplementedError: Cannot access storage of "
            "IpuTensorImpl")
    fine_tuning_harness(models.resnet18, trace_model)
