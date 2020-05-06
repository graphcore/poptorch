import torch
import torch.nn as nn
import numpy as np

import poptorch
import torchvision.models as models
import torchvision

from PIL import Image
from torchvision import transforms


training_batch_size = 1
training_ipu_step_size = 100
gradient_accumulation = 2
training_combined_batch_size = gradient_accumulation * training_batch_size * training_ipu_step_size


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = torchvision.datasets.CIFAR10('CIFAR10/', train=True, download=True,
                              transform=preprocess)

training_data = torch.utils.data.DataLoader(dataset, batch_size=training_combined_batch_size, shuffle=True, drop_last=True)

model = models.resnet18(pretrained=False)
model.train()

class printafterlayer(nn.Module):
  def __init__(self,layer):
    super(printafterlayer, self).__init__()
    self.layer = layer

  def __call__(self, x):
    print(self.layer)
    print("Input: " + str(x.size()))

    o = self.layer(x)

    print("Output: " + str(o.size()))
    return o



model.conv1 = printafterlayer(model.conv1)
model.bn1 = printafterlayer(model.bn1)
model.relu = printafterlayer(model.relu)

#model.layer3 = poptorch.IPU(1, model.layer3)

training_model = poptorch.trainingModel(model, training_ipu_step_size, gradient_accumulation=gradient_accumulation)

def train():
  for batch_number, (data, labels) in enumerate(training_data):
    result = training_model((data, labels.int()))

    if batch_number % 10 == 0:
        # Pick the highest probability.
        _, ind = torch.max(result, 1)
        eq = torch.eq(ind, labels)
        elms, counts = torch.unique(eq, sorted=False, return_counts=True)

        acc = 0.0
        if len(elms) == 2:
          if elms[0] == True:
            acc = (counts[0].item() / training_combined_batch_size) * 100.0
          else:
            acc = (counts[1].item()/ training_combined_batch_size) * 100.0

        print ("Training accuracy:  " + str(acc) +"% from batch of size " + str(training_combined_batch_size))


train()