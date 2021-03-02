#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm
import poptorch


def get_mnist_data(opts):
    options = poptorch.Options()
    training_data = poptorch.DataLoader(
        options,
        torchvision.datasets.MNIST('mnist_data/',
                                   train=True,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307, ), (0.3081, ))
                                   ])),
        batch_size=opts.batch_size * opts.batches_per_step,
        shuffle=True,
        drop_last=True)

    validation_data = poptorch.DataLoader(
        options,
        torchvision.datasets.MNIST('mnist_data/',
                                   train=False,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307, ), (0.3081, ))
                                   ])),
        batch_size=opts.test_batch_size,
        shuffle=True,
        drop_last=True)
    return training_data, validation_data


#annotations_start
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(784, 784)
        self.layer2 = nn.Linear(784, 784)
        self.layer3 = nn.Linear(784, 128)
        self.layer4 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = x.view(-1, 784)
        with poptorch.Block("B1"):
            x = self.layer1(x)
        with poptorch.Block("B2"):
            x = self.layer2(x)
        with poptorch.Block("B3"):
            x = self.layer3(x)
        with poptorch.Block("B4"):
            x = self.layer4(x)
            x = self.softmax(x)
        return x


class TrainingModelWithLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, args, loss_inputs=None):
        output = self.model(args)
        if loss_inputs is None:
            return output
        with poptorch.Block("B4"):
            loss = self.loss(output, loss_inputs)
        return output, loss


#annotations_end


def accuracy(predictions, labels):
    _, ind = torch.max(predictions, 1)
    # provide labels only for samples, where prediction is available (during the training, not every samples prediction is returned for efficiency reasons)
    labels = labels[-predictions.size()[0]:]
    accuracy = torch.sum(torch.eq(ind, labels)).item() / \
        labels.size()[0] * 100.0
    return accuracy


def train(training_model, training_data, opts):
    nr_batches = len(training_data)
    for epoch in range(1, opts.epochs + 1):
        print("Epoch {0}/{1}".format(epoch, opts.epochs))
        bar = tqdm(training_data, total=nr_batches)
        for data, labels in bar:
            preds, losses = training_model(data, labels)
            with torch.no_grad():
                mean_loss = torch.mean(losses).item()
                acc = accuracy(preds, labels)
            bar.set_description("Loss:{:0.4f} | Accuracy:{:0.2f}%".format(
                mean_loss, acc))
            if opts.profile:
                return


def test(inference_model, test_data):
    nr_batches = len(test_data)
    sum_acc = 0.0
    with torch.no_grad():
        for data, labels in tqdm(test_data, total=nr_batches):
            output = inference_model(data)
            sum_acc += accuracy(output, labels)
    print("Accuracy on test set: {:0.2f}%".format(sum_acc / len(test_data)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST training in PopTorch')
    parser.add_argument('--batch-size',
                        type=int,
                        default=4,
                        help='batch size for training (default: 4)')
    parser.add_argument('--batches-per-step',
                        type=int,
                        default=8,
                        help='device iteration (default:8)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=2,
                        help='batch size for testing (default: 4)')
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument(
        '--profile',
        type=str,
        help=
        "do a single iteration of training for profiling and place in a folder"
    )
    parser.add_argument('--strategy',
                        choices=['plain', 'pipelined', 'phased'],
                        default='plain',
                        help='execution strategy')
    parser.add_argument('--offload-opt',
                        type=bool,
                        help="offload optimizer state")
    opts = parser.parse_args()

    poptorch.setLogLevel(1)  # Force debug logging

    #annotations_strategy_start
    training_data, test_data = get_mnist_data(opts)
    model = Network()
    model_with_loss = TrainingModelWithLoss(model)
    model_opts = poptorch.Options().deviceIterations(1)
    if opts.strategy == "phased":
        strategy = poptorch.SerialPhasedExecution("B1", "B2", "B3", "B4")
        strategy.stage("B1").ipu(0)
        strategy.stage("B2").ipu(0)
        strategy.stage("B3").ipu(0)
        strategy.stage("B4").ipu(0)
        model_opts.setExecutionStrategy(strategy)
    elif opts.strategy == "pipelined":
        strategy = poptorch.PipelinedExecution("B1", "B2", "B3", "B4")
        strategy.stage("B1").ipu(0)
        strategy.stage("B2").ipu(1)
        strategy.stage("B3").ipu(2)
        strategy.stage("B4").ipu(3)
        model_opts.setExecutionStrategy(strategy)
        model_opts.Training.gradientAccumulation(opts.batches_per_step)
    else:
        strategy = poptorch.ShardedExecution("B1", "B2", "B3", "B4")
        strategy.stage("B1").ipu(0)
        strategy.stage("B2").ipu(0)
        strategy.stage("B3").ipu(0)
        strategy.stage("B4").ipu(0)
        model_opts.setExecutionStrategy(strategy)

    if opts.offload_opt:
        model_opts.TensorLocations.setActivationLocation(
            poptorch.TensorLocationSettings().useOnChipStorage(True))
        model_opts.TensorLocations.setWeightLocation(
            poptorch.TensorLocationSettings().useOnChipStorage(True))
        model_opts.TensorLocations.setAccumulatorLocation(
            poptorch.TensorLocationSettings().useOnChipStorage(True))
        model_opts.TensorLocations.setOptimizerLocation(
            poptorch.TensorLocationSettings().useOnChipStorage(False))

    training_model = poptorch.trainingModel(
        model_with_loss,
        model_opts,
        optimizer=optim.AdamW(model.parameters(), lr=opts.lr))

    # run training, on IPU
    train(training_model, training_data, opts)
    #annotations_strategy_end

    if opts.profile:
        sys.exit(1)

    # Update the weights in model by copying from the training IPU. This updates (model.parameters())
    training_model.copyWeightsToHost()

    # Check validation loss on IPU once trained. Because PopTorch will be compiled on first call the
    # weights in model.parameters() will be copied implicitly. Subsequent calls will need to call
    # inference_model.copyWeightsToDevice()
    inf_opts = poptorch.Options().deviceIterations(opts.test_batch_size)
    strategy = poptorch.ShardedExecution("B1", "B2", "B3", "B4")
    strategy.stage("B1").ipu(0)
    strategy.stage("B2").ipu(0)
    strategy.stage("B3").ipu(0)
    strategy.stage("B4").ipu(0)
    inf_opts.setExecutionStrategy(strategy)

    inference_model = poptorch.inferenceModel(model, inf_opts)
    test(inference_model, test_data)
