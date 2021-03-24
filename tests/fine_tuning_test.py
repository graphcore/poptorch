#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import copy
import os  # pylint: disable=unused-import
import unittest.mock
import torch
import torchvision.models as models
import helpers


def fine_tuning_harness(imagenet_model):
    torch.manual_seed(42)

    num_classes = 2
    num_examples = 2
    num_epochs = 20

    data = torch.randn((num_examples, 3, 224, 224))
    target = torch.randint(0, num_classes, (num_examples, ))

    model = imagenet_model(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False

    # Change the linear classifier at the top.
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    for param in model.fc.parameters():
        assert param.requires_grad
    initial_params = copy.deepcopy(model).state_dict()

    # Fine tune.
    optim = torch.optim.SGD(model.fc.parameters(), lr=0.001)
    poptorch_model = helpers.trainingModelWithLoss(model,
                                                   torch.nn.CrossEntropyLoss(),
                                                   optimizer=optim)

    for _ in range(num_epochs):
        _ = poptorch_model(data, target)

    # Assert only the last layer was changed.
    for name, param in model.named_parameters():
        if name.startswith('fc'):
            assert not torch.allclose(param.data, initial_params[name])
        else:
            helpers.assert_allclose(actual=param.data,
                                    expected=initial_params[name])


@unittest.mock.patch.dict("os.environ", helpers.disableSmallModel())
def test_resnet18():
    fine_tuning_harness(models.resnet18)
