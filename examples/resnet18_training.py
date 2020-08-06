#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch
import torchvision.models as models
import torchvision

from torchvision import transforms

training_batch_size = 1
training_ipu_step_size = 100
gradient_accumulation = 2

opts = poptorch.Options().deviceIterations(training_ipu_step_size)
opts.Training.gradientAccumulation(gradient_accumulation)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                          0.225]),
])

dataset = torchvision.datasets.CIFAR10('CIFAR10/',
                                       train=True,
                                       download=True,
                                       transform=preprocess)

training_data = poptorch.DataLoader(opts,
                                    dataset,
                                    batch_size=training_batch_size,
                                    shuffle=True,
                                    drop_last=True)


class NormWrapper(torch.nn.Module):
    def __init__(self, C):
        super(NormWrapper, self).__init__()
        self.layer = torch.nn.GroupNorm(1, C)

    def forward(self, *input, **kwargs):
        return self.layer(*input, **kwargs)


model = models.resnet18(pretrained=False, norm_layer=NormWrapper)
model.train()
training_model = poptorch.trainingModel(model, opts, loss=torch.nn.NLLLoss())


def train():
    for batch_number, (data, labels) in enumerate(training_data):
        result, _ = training_model(data, labels)
        if batch_number == 0:
            print("Model compiled, training...", flush=True)

        if batch_number % 10 == 0:
            # Pick the highest probability.
            _, ind = torch.max(result, 1)
            eq = torch.eq(ind, labels)
            elms, counts = torch.unique(eq, sorted=False, return_counts=True)

            acc = 0.0
            if len(elms) == 2:
                if elms[0]:
                    acc = (counts[0].item() /
                           training_data.combinedBatchSize) * 100.0
                else:
                    acc = (counts[1].item() /
                           training_data.combinedBatchSize) * 100.0

            print("Training accuracy:  " + str(acc) + "% from batch of size " +
                  str(training_data.combinedBatchSize),
                  flush=True)


train()
