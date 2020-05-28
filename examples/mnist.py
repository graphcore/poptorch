import torch
import torch.nn as nn
import torchvision
import numpy as np
import poptorch

# Load the MNIST data.

training_batch_size = 20
training_ipu_step_size = 20

training_combined_batch_size = training_batch_size * training_ipu_step_size
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
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
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
        # with poptorch.IPU(0):
        x = self.layer1(x)
        x = self.layer2(x)
        # -1 means deduce from the above layers, this is just batch size for most iterations.
        x = x.view(-1, 320)

        # with poptorch.IPU(1):
        x = self.layer3_act(self.layer3(x))
        x = self.layer4(x)
        #x = poptorch.ipu_print_tensor(x)
        return self.softmax(x)


# Create our model.
model = Network()

# This isn't needed. It's just to print out the loss in python land.
loss_function = nn.CrossEntropyLoss()

# Create model for training which will run on IPU.
training_model = poptorch.trainingModel(model, training_ipu_step_size)

# Same model as above, they will share weights (in 'model') so while the above
# trains the weights in the weights in this will automatically update.
inference_model = poptorch.inferenceModel(model)


def train():
    for batch_number, (data, labels) in enumerate(training_data):
        result = training_model((data, labels.int()))
        popart_loss = loss_function(result, labels)

        if batch_number % 10 == 0:
            print("PoptorchIPU loss at batch: " + str(batch_number) + " is " +
                  str(popart_loss))

            # Pick the highest probability.
            _, ind = torch.max(result, 1)
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


train()
test()
