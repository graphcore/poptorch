# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import poptorch


class ExampleModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.device = device

    def forward(self, x):
        # The following tensor is created explicitly inside the model, so its
        # device has to be specified.
        # Alternatively, the user can set option 'forceAllTensorsDeviceToIpu'
        # in poptorch.Options() which will set tensors device to ipu by default.
        model_tensor = torch.tensor([0], device=self.device)
        return model_tensor + torch.cat([
            100 * torch.nn.LeakyReLU()(-x + self.bias),
            100 * torch.nn.LeakyReLU()(x - self.bias)
        ],
                                        dim=-1)


# model_with_loss_start
class ExampleModelWithLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = ExampleModel(device)

    def forward(self, input, target):
        out = self.model(input)

        return (torch.nn.functional.softmax(out),
                torch.nn.CrossEntropyLoss(reduction="mean")(out, target))


# model_with_loss_end


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
            data = (torch.rand(self._shape) + label) * 0.5
            self._all_data.append(data)
            self._all_labels.append(label)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self._all_data[index], self._all_labels[index]


def run_examples():
    # simple_ipu_start
    # Set up the PyTorch DataLoader to load that much data at each iteration
    opts = poptorch.Options()
    opts.deviceIterations(10)
    training_data = poptorch.DataLoader(options=opts,
                                        dataset=ExampleDataset(shape=[1],
                                                               length=20000),
                                        batch_size=10,
                                        shuffle=True,
                                        drop_last=True)

    model = ExampleModelWithLoss(device='ipu')
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Wrap the model in a PopTorch training wrapper
    poptorch_model = poptorch.trainingModel(model,
                                            options=opts,
                                            optimizer=optimizer)

    momentum_loss = None

    for batch, target in training_data:
        # Performs forward pass, loss function evaluation,
        # backward pass and weight update in one go on the device.
        _, loss = poptorch_model(batch, target)

        if momentum_loss is None:
            momentum_loss = loss
        else:
            momentum_loss = momentum_loss * 0.95 + loss * 0.05

        # Optimizer can be updated via setOptimizer.
        if momentum_loss < 0.1:
            poptorch_model.setOptimizer(
                torch.optim.AdamW(model.parameters(), lr=0.0001))
    # simple_ipu_end

    assert (model.model.bias > 0.4 and model.model.bias < 0.6)

    # simple_cpu_start
    training_data = torch.utils.data.DataLoader(ExampleDataset(shape=[1],
                                                               length=20000),
                                                batch_size=10,
                                                shuffle=True,
                                                drop_last=True)

    model = ExampleModelWithLoss(device='cpu')
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    momentum_loss = None

    for batch, target in training_data:
        # Zero gradients
        optimizer.zero_grad()

        # Run model.
        _, loss = model(batch, target)

        # Back propagate the gradients.
        loss.backward()

        # Update the weights.
        optimizer.step()

        if momentum_loss is None:
            momentum_loss = loss
        else:
            momentum_loss = momentum_loss * 0.95 + loss * 0.05

        if momentum_loss < 0.1:
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    # simple_cpu_end

    print(model.model.bias)
    assert (model.model.bias > 0.4 and model.model.bias < 0.6)


if __name__ == "__main__":
    run_examples()
