#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch
import torch.optim as optim


def test_layerNorm():
    torch.manual_seed(42)

    for i in range(1, 4):
        input = torch.randn([3, 2, 5, 2])
        layerNorm = torch.nn.LayerNorm(input.size()[i:])

        # Run pytorch native on CPU.
        nativeOutput = layerNorm(input)

        # Run on IPU.
        ipuModel = poptorch.inferenceModel(layerNorm)
        poptorchOut = ipuModel(input)

        assert torch.allclose(poptorchOut, nativeOutput)


def test_layerNormScalar():
    torch.manual_seed(42)

    input = torch.randn([3, 2, 5, 2])
    layerNorm = torch.nn.LayerNorm(2)

    # Run pytorch native on CPU.
    nativeOutput = layerNorm(input)

    # Run on IPU.
    ipuModel = poptorch.inferenceModel(layerNorm)
    poptorchOut = ipuModel(input)

    assert torch.allclose(poptorchOut, nativeOutput)


def test_layerNormPretrainedWeights():
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = torch.nn.Conv2d(5, 5, kernel_size=(1, 1))
            self.norm = torch.nn.LayerNorm((5, 3, 10))

        def forward(self, x):
            x = self.conv(x)

            return self.norm(x)

    model = Model()

    input = torch.randn([3, 5, 3, 10])

    modelOut = model(input)

    # Run on IPU.
    ipuModel = poptorch.inferenceModel(model)
    poptorchOut = ipuModel(input)

    # Marginally more leeway.
    assert torch.allclose(poptorchOut, modelOut, rtol=1e-4, atol=1e-6)

    # We aren't training to any real target we just want to update the beta/gamma parameters and check they still work in popart.
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print(list(model.norm.parameters()))

    model.train()
    for i in range(0, 10):
        outputs = model(input)
        otimizer.zero_grad()
        loss = criterion(outputs, torch.ones([3, 5, 3, 10]))
        loss.backward()
        optimizer.step()

    model.eval()
    # Run on IPU with trained weights.
    ipuModel = poptorch.inferenceModel(model)
    poptorchOut = ipuModel(input)

    # Run on CPU again with trained weights.
    outputs = model(input)

    assert torch.allclose(poptorchOut, outputs, rtol=1e-4, atol=1e-6)
