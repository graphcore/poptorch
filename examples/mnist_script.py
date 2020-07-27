#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import readline
import torch
import torch.nn as nn
import torch.functional
import torchvision
import numpy as np
import poptorch

readline.parse_and_bind('tab:complete')

# Load the MNIST data.
validation_batch_size = 100

validation_data = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
    'mnist_data/',
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
    ])),
                                              batch_size=validation_batch_size,
                                              shuffle=True,
                                              drop_last=True)


# Define the network.
class Block(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, pool_size):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              num_filters,
                              kernel_size=kernel_size)
        self.batch = nn.BatchNorm2d(num_filters)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.pool(x)
        x = self.relu(x)
        return x


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = Block(1, 10, 5, 2)
        self.layer2 = Block(10, 20, 5, 2)
        self.layer3 = nn.Linear(320, 256)
        self.layer3_act = nn.ReLU()
        self.layer4 = nn.Linear(256, 10)

        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # -1 means deduce from the above layers, this is just batch size for most iterations.
        x = x.view(-1, 320)

        x = self.layer3_act(self.layer3(x))
        x = self.layer4(x)
        return self.softmax(x)


d0, _ = next(iter(validation_data))

# Create our model.
model = Network()
model.eval()

opts = poptorch.Options().deviceIterations(4)
opts.Jit.traceModel(False)
inference_model = poptorch.inferenceModel(model, opts)
result = inference_model(d0)
result = result.numpy()

ref = model(d0)
ref = ref.detach().numpy()

if not np.allclose(result, ref):
    print('ERROR: arrays are not close')
else:
    print('Success: arrays are close')
