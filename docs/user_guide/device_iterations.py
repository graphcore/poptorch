# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import sys
import poptorch

if not poptorch.ipuHardwareIsAvailable():
    print("Replicated top level graphs are not supported on the IPU model")
    sys.exit(0)
# pylint: disable=unused-variable, wrong-import-position, reimported, ungrouped-imports, wrong-import-order
# iterations_start
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
        super().__init__()
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


def device_iterations_example():
    # Set the number of samples for which activations/gradients are computed
    # in parallel on a single IPU
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

    # Run over the training data, 100 batches at a time (specified in
    # opts.deviceIterations())
    for batch_number, (data, labels) in enumerate(training_data):
        # Execute the device with a 100 iteration loop of batchsize 2.
        # "output" and "loss" will be the respective output and loss of the
        # final batch (the default OutputMode).

        output, loss = poptorch_model(data, labels)
        print(f"{labels[-1]}, {output}, {loss}")
    # iterations_end
    poptorch_model.destroy()  # release the IPUs


# replication_start
def replication_factor_example():
    # Set the number of samples for which activations/gradients are computed
    # in parallel on a single IPU
    model_batch_size = 2
    # replication_start
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

    # Run over the training data, 100 batches at a time (specified in
    # opts.deviceIterations())
    for batch_number, (data, labels) in enumerate(training_data):
        # Execute the device with a 100 iteration loop of model batchsize 2
        # across 4 IPUs (global batchsize = 2 * 4 = 8). "output" and "loss"
        # will be the respective output and loss of the final batch of each
        # replica (the default OutputMode).
        output, loss = poptorch_model(data, labels)
        print(f"{labels[-1]}, {output}, {loss}")
    # replication_end
    poptorch_model.destroy()  # release the IPUs


# gradient_acc_start
def gradient_accumulation_example():
    # Set the number of samples for which activations/gradients are computed
    # in parallel on a single IPU
    model_batch_size = 2
    # Create a poptorch.Options instance to override default options
    opts = poptorch.Options()

    # Run a 400 iteration loop on the IPU, fetching a new batch each time
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

    # Run over the training data, 400 batches at a time (specified in
    # opts.deviceIterations())
    for batch_number, (data, labels) in enumerate(training_data):
        # Execute the device with a 100 iteration loop of model batchsize 2
        # with gradient updates every 8 iterations (global batchsize = 2 * 8 = 16).
        # "output" and "loss" will be the respective output and loss of the
        # final batch of each replica (the default OutputMode).
        output, loss = poptorch_model(data, labels)
        print(f"{labels[-1]}, {output}, {loss}")
    # gradient_acc_end
    poptorch_model.destroy()  # release the IPUs


def data_accessor_example():
    # Not displayed: just to keep the linter happy
    shape = [3, 2]
    num_tensors = 100
    batch_size = 1
    num_workers = 0
    device_iterations = 1
    replication_factor = 1
    # Example starts here:
    # data_accessor_start
    opts = poptorch.Options()
    opts.deviceIterations(device_iterations)
    opts.replicationFactor(replication_factor)

    loader = poptorch.DataLoader(opts,
                                 ExampleDataset(shape=shape,
                                                length=num_tensors),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 mode=poptorch.DataLoaderMode.Async)

    poptorch_model = poptorch.inferenceModel(model, opts)

    for it, (data, _) in enumerate(loader):
        out = poptorch_model(data)
    # data_accessor_end
    poptorch_model.destroy()  # release the IPUs


model = ExampleModelWithLoss(data_shape=[3, 2], num_classes=2)
model_batch_size = 2


# distributed_execution_start
def process(process_id=0, num_processes=1):
    # Create a poptorch.Options instance to override default options
    opts = poptorch.Options()

    # Run a 100 iteration loop on the IPU, fetching a new batch each time
    opts.deviceIterations(400)

    # Replicate the graph across 2 IPUs in each process.
    opts.replicationFactor(2)

    # Set the id of the current process and the total number of processes.
    opts.Distributed.configureProcessId(process_id, num_processes)

    # Accumulate the gradient 8 times before applying it.
    opts.Training.gradientAccumulation(8)

    # Optional: All the processes must use the same seed if shuffle=True is used for the DataLoader.
    opts.randomSeed(42)

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
        # Execute the device with a 100 iteration loop of batchsize 8 across
        # 4 IPUs (batch-size 2 per replica). "output" and "loss" will be the
        # respective output and loss of the final batch of each replica
        # (the default OutputMode).
        output, loss = poptorch_model(data, labels)
        print(f"{batch_number} {labels[-1]}, {output}, {loss}")

    # distributed_execution_end
    poptorch_model.destroy()  # release the IPUs


# AsynchronousDataAccessor must run in the main process
if __name__ == "__main__":
    device_iterations_example()
    replication_factor_example()
    gradient_accumulation_example()
    data_accessor_example()
    process()
