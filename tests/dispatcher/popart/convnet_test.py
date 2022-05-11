#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import copy
import pytest
import torch
import torch.nn as nn
from poptorch.experimental import IPUContext
import helpers
import poptorch


@pytest.mark.mlirSupportRequired
def test_mnist():
    # A helper block to build convolution-pool-relu blocks.
    class Block(nn.Module):
        def __init__(self, in_channels, num_filters, kernel_size, pool_size):
            super(Block, self).__init__()
            self.conv = nn.Conv2d(in_channels,
                                  num_filters,
                                  kernel_size=kernel_size)
            self.pool = nn.MaxPool2d(kernel_size=pool_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = self.relu(x)
            return x

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = Block(1, 10, 5, 2)
            self.layer2 = Block(10, 20, 5, 2)
            self.layer3 = nn.Linear(320, 256, False)
            self.layer3_act = nn.ReLU()
            self.layer4 = nn.Linear(256, 10)

            self.softmax = nn.LogSoftmax(1)
            self.loss = nn.NLLLoss(reduction="mean")

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)

            x = x.view(-1, 320)

            x = self.layer3_act(self.layer3(x))

            x = self.layer4(x)
            x = self.softmax(x)
            return x

    model = Network()
    input = torch.ones([1, 1, 28, 28])

    cpu_model = copy.deepcopy(model)
    ipu_out = IPUContext(model, model=model,
                         compiler=poptorch.Compiler.PopART)(input)

    helpers.assert_allclose(expected=cpu_model(input),
                            actual=ipu_out,
                            atol=1e-05,
                            rtol=1e-05,
                            equal_nan=True)
