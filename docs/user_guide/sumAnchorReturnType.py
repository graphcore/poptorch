# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import random
import torch
import poptorch

RAND_SEED = 8549


class ExampleClassDataset(torch.utils.data.Dataset):
    """ A dummy dataset with classes for emulating a classification task.

    All instances of a class, C, will correspond to R*V where
    R is a randomly generated rotation matrix, fixed for the whole dataset
    V = V_all + V_cls
    V_all is a vector of vec_length for which all elements are sampled
    from an i.i.d. normal distribution, V_all ~ N(0, 0.2).
    V_cls is a vector of vec_length such that
    V_cls[x] ~ N(1, 0.2), if x = C, (i.e. the class label)
             = 0, otherwise

    """

    def __init__(self, num_classes, vec_length, num_examples):
        super().__init__()
        assert vec_length >= num_classes

        random.seed(RAND_SEED)

        #Generate the class label at this point
        self.targets = [None] * num_examples
        for idx in range(num_examples):
            self.targets[idx] = random.randrange(num_classes)

        # To get R, make a random symmetric matrix and use eigenvalue
        # decomposition
        torch.manual_seed(RAND_SEED)
        R = torch.rand([vec_length, vec_length])
        R = R + R.transpose(0, 1)
        self._R = torch.eig(R, eigenvectors=True).eigenvectors

        # # For now, use identity for R
        # self._R = torch.eye(vec_length, vec_length)

        self._dist = torch.distributions.normal.Normal(0, 0.2)
        self._dist = self._dist.expand([vec_length])

        self._vec_length = vec_length

    def __getitem__(self, idx):
        torch.manual_seed(idx + RAND_SEED)
        v = self._dist.sample()
        item_cls = self.targets[idx]
        v[item_cls] += 1.0

        v = torch.matmul(self._R, v)

        return v, item_cls

    def __len__(self):
        return len(self.targets)


# yapf: disable
#model_returning_accuracy_start
class MulticlassPerceptron(torch.nn.Module):
    def __init__(self, vec_length, num_classes):
        super().__init__()
        self.fc = torch.nn.Linear(vec_length, num_classes)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, target):
        fc = self.fc(x)

        classification = torch.argmax(fc, dim=-1)
        accuracy = (torch.sum((classification == target).to(torch.float)) /
                    float(classification.numel()))

        if self.training:
            return self.loss(fc, target), accuracy

        return classification, accuracy
# model_returning_accuracy_end
# yapf: enable

NUM_CLASSES = 10
VEC_LENGTH = NUM_CLASSES * 2

# yapf: disable
#sum_accuracy_start
opts = poptorch.Options()

opts.deviceIterations(5)
opts.Training.gradientAccumulation(10)
opts.outputMode(poptorch.OutputMode.Sum)

training_data = poptorch.DataLoader(opts,
                                    dataset=ExampleClassDataset(
                                        NUM_CLASSES, VEC_LENGTH, 2000),
                                    batch_size=5,
                                    shuffle=True,
                                    drop_last=True)


model = MulticlassPerceptron(VEC_LENGTH, NUM_CLASSES)
model.train()

# Wrap the model in a PopTorch training wrapper
poptorch_model = poptorch.trainingModel(model,
                                        options=opts,
                                        optimizer=torch.optim.Adam(
                                            model.parameters()))

# Run over the training data, 5 batches at a time.
for batch_number, (data, labels) in enumerate(training_data):
    # Execute the device with a 5 iteration loop of batchsize 5 with 10
    # gradient accumulations (global batchsize = 5 * 10 = 50). "loss" and
    # "accuracy" will be the sum across all device iterations and gradient
    # accumulations but not across the model batch size.
    _, accuracy = poptorch_model(data, labels)

    # Correct for iterations
    # Do not divide by batch here, as this is already accounted for in the
    # PyTorch Model.
    accuracy /= (opts.device_iterations * opts.Training.gradient_accumulation)
    print(f"Accuracy: {float(accuracy)*100:.2f}%")
#sum_accuracy_end
# yapf: enable
