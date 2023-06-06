# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from copy import deepcopy
import pytest
import torch

from torch_geometric.nn import Linear
from poptorch import inferenceModel, trainingModel


class MishReference(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class MishTrainModel(torch.nn.Module):
    def __init__(self, op, linear):
        super().__init__()
        self.loss_fn = torch.nn.MSELoss()
        self.linear = linear
        self.op = op

    def forward(self, x):
        result = self.op(x)
        res = result.float()
        result = self.linear(res)
        target = torch.ones_like(result)
        loss = self.loss_fn(result, target)
        return result, loss


@pytest.mark.parametrize('size', [(13, ), (1, 64, 320, 320)])
def test_mish(size):
    x = torch.rand(size)
    ipu_model = inferenceModel(torch.nn.Mish())
    ipu_res = ipu_model(x)

    ref_ipu_model = inferenceModel(MishReference())
    ref_ipu_res = ref_ipu_model(x)

    ref_model = torch.nn.Mish()
    ref_res = ref_model(x)

    torch.allclose(ipu_res, ref_ipu_res)
    torch.allclose(ipu_res, ref_res)


@pytest.mark.parametrize('size', [(11, ), (1, 64, 128)])
def test_mish_training(size):
    x = torch.rand(size)
    linear_ipu = Linear(size[-1], size[-1])
    linear_ref = deepcopy(linear_ipu)
    model = MishTrainModel(torch.nn.Mish(), linear_ipu)

    ref_res, ref_loss = model(x)

    ipu_model = trainingModel(model)
    ipu_res, ipu_loss = ipu_model(x)

    model = MishTrainModel(MishReference(), linear_ref)
    ref_ipu_model = trainingModel(model)
    ref_ipu_res, ref_ipu_loss = ref_ipu_model(x)

    torch.allclose(ipu_res, ref_res)
    torch.allclose(ipu_loss, ref_loss)
    torch.allclose(ipu_res, ref_ipu_res)
    torch.allclose(ipu_loss, ref_ipu_loss)
