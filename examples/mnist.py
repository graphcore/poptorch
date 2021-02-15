#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.


# pylint: disable=too-many-statements
def example():
    # pylint: disable=import-outside-toplevel
    import sys
    import poptorch
    if not poptorch.ipuHardwareIsAvailable():
        poptorch.logger.warn("This examples requires IPU hardware to run")
        sys.exit(0)

    # pylint: disable=unused-variable, wrong-import-position, reimported, ungrouped-imports, wrong-import-order, import-outside-toplevel
    # mnist_start
    import torch
    import torch.nn as nn
    import torchvision
    import poptorch

    # Normal pytorch batch size
    training_batch_size = 20
    validation_batch_size = 100

    opts = poptorch.Options()
    # Device "step"
    opts.deviceIterations(20)

    # How many IPUs to replicate over.
    opts.replicationFactor(4)

    opts.randomSeed(42)

    # Load MNIST normally.
    training_data = poptorch.DataLoader(
        opts,
        torchvision.datasets.MNIST('mnist_data/',
                                   train=True,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307, ), (0.3081, ))
                                   ])),
        batch_size=training_batch_size,
        shuffle=True)

    # Load MNIST normally.
    val_options = poptorch.Options()
    validation_data = poptorch.DataLoader(
        val_options,
        torchvision.datasets.MNIST('mnist_data/',
                                   train=True,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307, ), (0.3081, ))
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
            super().__init__()
            self.layer1 = Block(1, 10, 5, 2)
            self.layer2 = Block(10, 20, 5, 2)
            self.layer3 = nn.Linear(320, 256)
            self.layer3_act = nn.ReLU()
            self.layer4 = nn.Linear(256, 10)

            self.softmax = nn.LogSoftmax(1)
            self.loss = nn.NLLLoss(reduction="mean")

        def forward(self, x, target=None):
            x = self.layer1(x)
            x = self.layer2(x)
            x = x.view(-1, 320)

            x = self.layer3_act(self.layer3(x))
            x = self.layer4(x)
            x = self.softmax(x)

            if target is not None:
                loss = self.loss(x, target)
                return x, loss
            return x

    # Create our model.
    model = Network()

    # Create model for training which will run on IPU.
    training_model = poptorch.trainingModel(model, training_data.options)

    # Same model as above, they will share weights (in 'model') which once training is finished can be copied back.
    inference_model = poptorch.inferenceModel(model, validation_data.options)

    def train():
        for batch_number, (data, labels) in enumerate(training_data):
            output, losses = training_model(data, labels)

            if batch_number % 10 == 0:
                print(f"PoptorchIPU loss at batch: {batch_number} is {losses}")

                # Pick the highest probability.
                _, ind = torch.max(output, 1)
                assert training_data.options.anchor_mode in (
                    poptorch.AnchorMode.All, poptorch.AnchorMode.Final
                ), "Only 'Final' and 'All' AnchorMode supported"
                # If we're using Final: only keep the last labels, no-op if using All
                num_labels = ind.shape[0]
                labels = labels[-num_labels:]
                eq = torch.eq(ind, labels)
                elms, counts = torch.unique(eq,
                                            sorted=False,
                                            return_counts=True)

                acc = 0.0
                if len(elms) == 2:
                    if elms[0]:
                        acc = (counts[0].item() / num_labels) * 100.0
                    else:
                        acc = (counts[1].item() / num_labels) * 100.0

                print(
                    f"Training accuracy: {acc}% from batch of size {num_labels}"
                )
        print("Done training")

    def test():
        correct = 0
        total = 0
        with torch.no_grad():
            for (data, labels) in validation_data:
                output = inference_model(data)

                # Argmax the probabilities to get the highest.
                _, ind = torch.max(output, 1)

                # Compare it against the ground truth for this batch.
                eq = torch.eq(ind, labels)

                # Count the number which are True and the number which are False.
                elms, counts = torch.unique(eq,
                                            sorted=False,
                                            return_counts=True)

                if len(elms) == 2 or elms[0]:
                    if elms[0]:
                        correct += counts[0].item()
                    else:
                        correct += counts[1].item()

                total += validation_batch_size
        print("Validation: of " + str(total) + " samples we got: " +
              str((correct / total) * 100.0) + "% correct")

    # Train on IPU.
    train()

    test()
    # mnist_end


# AsynchronousDataAccessor must run in the main process
if __name__ == "__main__":
    example()
