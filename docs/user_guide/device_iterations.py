# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from functools import reduce
from operator import mul

import torch
import poptorch


class ExampleModelWithLoss(torch.nn.Module):
    def __init__(self, data_shape, num_classes):
        super().__init__()

        self.fc = torch.nn.Linear(reduce(mul, data_shape), num_classes)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, target=None):
        reshaped = x.reshape([x.shape[0], -1])
        fc = self.fc(reshaped)

        if target is not None:
            return fc, self.loss(fc, target)
        return fc


class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, shape, length):
        self._shape = shape
        self._length = length

        self._all_data = []
        self._all_labels = []

        torch.manual_seed(0)
        for _ in range(length):
            label = 1 if torch.rand(()) > 0.5 else 0
            data = torch.rand(self._shape) + label
            data[0] = -data[0]
            self._all_data.append(data)
            self._all_labels.append(label)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self._all_data[index], self._all_labels[index]


# Set the batch size in the conventional sense of being the size that
# runs through an operation in the model at any given time
model_batch_size = 2

# Create a poptorch.Options instance to override default options
opts = poptorch.Options()

# Run a 100 iteration loop on the IPU, fetching a new batch each time
opts.deviceIterations(100)

# Set up the DataLoader to load that much data at each iteration
training_data = poptorch.DataLoader(opts,
                                    dataset=ExampleDataset(shape=[3, 2],
                                                           length=10000),
                                    batch_size=model_batch_size,
                                    shuffle=True,
                                    drop_last=True)

model = ExampleModelWithLoss(data_shape=[3, 2], num_classes=2)
# Wrap the model in a PopTorch training wrapper
poptorch_model = poptorch.trainingModel(model, options=opts)

# Run over the training data with "batch_size" 200 essentially.
for batch_number, (data, labels) in enumerate(training_data):
    # Execute the device with a 100 iteration loop of batchsize 2.
    # "output" and "loss" will be the respective output and loss of the final
    # batch (the default AnchorMode).

    output, loss = poptorch_model(data, labels)
    print(f"{labels[-1]}, {output}, {loss}")

model_batch_size = 2

# Create a poptorch.Options instance to override default options
opts = poptorch.Options()

# Run a 100 iteration loop on the IPU, fetching a new batch each time
opts.deviceIterations(100)

# Duplicate the model over 4 replicas.
opts.replicationFactor(4)

training_data = poptorch.DataLoader(opts,
                                    dataset=ExampleDataset(shape=[3, 2],
                                                           length=100000),
                                    batch_size=model_batch_size,
                                    shuffle=True,
                                    drop_last=True)

model = ExampleModelWithLoss(data_shape=[3, 2], num_classes=2)
# Wrap the model in a PopTorch training wrapper
poptorch_model = poptorch.trainingModel(model, options=opts)

# Run over the training data with "batch_size" 200 essentially.
for batch_number, (data, labels) in enumerate(training_data):
    # Execute the device with a 100 iteration loop of batchsize 2 across
    # 4 IPUs. "output" and "loss" will be the respective output and loss of the
    # final batch of each replica (the default AnchorMode).
    output, loss = poptorch_model(data, labels)
    print(f"{labels[-1]}, {output}, {loss}")

# Create a poptorch.Options instance to override default options
opts = poptorch.Options()

# Run a 100 iteration loop on the IPU, fetching a new batch each time
opts.deviceIterations(400)

# Accumulate the gradient 8 times before applying it.
opts.Training.gradientAccumulation(8)

training_data = poptorch.DataLoader(opts,
                                    dataset=ExampleDataset(shape=[3, 2],
                                                           length=100000),
                                    batch_size=model_batch_size,
                                    shuffle=True,
                                    drop_last=True)

# Wrap the model in a PopTorch training wrapper
poptorch_model = poptorch.trainingModel(model, options=opts)

# Run over the training data with "batch_size" 200 essentially.
for batch_number, (data, labels) in enumerate(training_data):
    # Execute the device with a 100 iteration loop of batchsize 2 across
    # 4 IPUs. "output" and "loss" will be the respective output and loss of the
    # final batch of each replica (the default AnchorMode).
    output, loss = poptorch_model(data, labels)
    print(f"{labels[-1]}, {output}, {loss}")
