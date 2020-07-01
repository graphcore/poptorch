#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import torchvision
import numpy as np
import poptorch

# Normal pytorch batch size
training_batch_size = 20

# Device "step"
training_ipu_step_size = 20

# How many IPUs to replicate over.
replication_factor = 4

# This is the amount of data we will pull out of the data loader at each step. This is not
# how much will be running on the IPU in a single model batch however. We just give the IPUs
# this much data to allow for more efficient data loading.
training_combined_batch_size = training_batch_size * training_ipu_step_size * replication_factor

# Load MNIST normally.
training_data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/',
                               train=True,
                               download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307, ),
                                                                    (0.3081, ))
                               ])),
    batch_size=training_combined_batch_size,
    shuffle=True,
    drop_last=True)

# Load MNIST normally. 100 is actually just the model batchsize this time.
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


# Define the network using the above blocks.
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = Block(1, 10, 5, 2)
        self.layer2 = Block(10, 20, 5, 2)
        self.layer3 = nn.Linear(320, 256)
        self.layer3_act = nn.ReLU()
        self.layer4 = nn.Linear(256, 10)

        self.softmax = nn.LogSoftmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 320)

        x = self.layer3_act(self.layer3(x))
        x = self.layer4(x)
        x = self.softmax(x)
        return x


# Create our model.
model = Network()

# Create model for training which will run on IPU.
opts = poptorch.Options().deviceIterations(training_ipu_step_size)
training_model = poptorch.trainingModel(model,
                                        opts,
                                        replication_factor=replication_factor,
                                        loss=nn.NLLLoss(reduction="mean"))

# Same model as above, they will share weights (in 'model') which once training is finished can be copied back.
inference_model = poptorch.inferenceModel(model)


def train():
    for batch_number, (data, labels) in enumerate(training_data):
        result = training_model(data, labels)

        if batch_number % 10 == 0:
            print("PoptorchIPU loss at batch: " + str(batch_number) + " is " +
                  str(result[1]))

            # Pick the highest probability.
            _, ind = torch.max(result[0], 1)
            eq = torch.eq(ind, labels)
            elms, counts = torch.unique(eq, sorted=False, return_counts=True)

            acc = 0.0
            if len(elms) == 2:
                if elms[0] == True:
                    acc = (counts[0].item() /
                           training_combined_batch_size) * 100.0
                else:
                    acc = (counts[1].item() /
                           training_combined_batch_size) * 100.0

            print("Training accuracy:  " + str(acc) + "% from batch of size " +
                  str(training_combined_batch_size))


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_number, (data, labels) in enumerate(validation_data):
            output = inference_model(data)

            # Argmax the probabilities to get the highest.
            _, ind = torch.max(output, 1)

            # Compare it against the ground truth for this batch.
            eq = torch.eq(ind, labels)

            # Count the number which are True and the number which are False.
            elms, counts = torch.unique(eq, sorted=False, return_counts=True)

            if len(elms) == 2 or elms[0] == True:
                if elms[0] == True:
                    correct += counts[0].item()
                else:
                    correct += counts[1].item()

            total += validation_batch_size
    print("Validation: of " + str(total) + " samples we got: " +
          str((correct / total) * 100.0) + "% correct")


# Train on IPU.
train()

# Update the weights in model by copying from the training IPU. This updates (model.parameters())
training_model.copyWeightsToHost()

# Check validation loss on IPU once trained. Because PopTorch will be compiled on first call the
# weights in model.parameters() will be copied implicitly. Subsequent calls will need to call
# inference_model.copyWeightsToDevice()
test()
