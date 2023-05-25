# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from copy import deepcopy
import numpy as np
import torch
import torch_cluster
from torch_geometric.nn import Linear
import pytest
import poptorch


class FpsInferModel(torch.nn.Module):
    def forward(self, x, ptr, ratio, random_start):
        return poptorch.fps(x, ptr, ratio, random_start)


class FpsTrainModel(torch.nn.Module):
    def __init__(self, op, linear):
        super().__init__()
        self.loss_fn = torch.nn.MSELoss()
        self.linear = linear
        self.op = op

    def forward(self, x, ptr, ratio, random_start):
        result = self.op(x, ptr, ratio, random_start)
        res = result.float()
        result = self.linear(res)
        target = torch.ones_like(result)
        loss = self.loss_fn(result, target)
        return result, loss


@pytest.mark.parametrize('src_shape', [(1, 2), (2, 19), (3, 10), (19, 3)])
@pytest.mark.parametrize('ratio', [0.3, 0.5, 1.0])
def test_single_batch(src_shape, ratio):
    src = torch.rand(src_shape)
    ptr = [0, src_shape[0]]
    batch = torch.zeros(src_shape[0], dtype=torch.long)

    inference_model = poptorch.inferenceModel(FpsInferModel())
    ipu_res = inference_model(src, ptr, ratio, random_start=False)
    ref_res = torch_cluster.fps(src, batch, ratio, random_start=False)

    assert all(ipu_res == ref_res)


@pytest.mark.parametrize('src_shape', [(19, 3)])
@pytest.mark.parametrize(
    'ptr', [[0, 13, 19], [0, 2, 3, 4, 9, 11, 19], [0, 1, 3, 4, 9, 18, 19]])
@pytest.mark.parametrize('ratio', [0.4, 0.6, 1.0])
def test_multi_batch(src_shape, ptr, ratio):
    src = torch.rand(src_shape)
    batch = torch.zeros(src_shape[0], dtype=torch.long)
    for i in range(1, len(ptr)):
        batch[ptr[i - 1]:ptr[i]] = i - 1

    inference_model = poptorch.inferenceModel(FpsInferModel())
    ipu_res = inference_model(src, ptr, ratio, random_start=False)
    ref_res = torch_cluster.fps(src, batch, ratio, random_start=False)

    assert all(ipu_res == ref_res)


@pytest.mark.parametrize('src_shape', [(29, 3)])
@pytest.mark.parametrize('ptr', [[0, 29], [0, 2, 6, 11, 28, 29]])
@pytest.mark.parametrize('ratio', [1.0])
def test_random_start(src_shape, ptr, ratio):
    src = torch.rand(src_shape)
    batch = torch.zeros(src_shape[0], dtype=torch.long)
    for i in range(1, len(ptr)):
        batch[ptr[i - 1]:ptr[i]] = i - 1

    inference_model = poptorch.inferenceModel(FpsInferModel())
    ipu_res = inference_model(src, ptr, ratio, random_start=True)
    ref_res = torch_cluster.fps(src, batch, ratio, random_start=True)

    for i in range(1, len(ptr)):
        ipu_res_slice = set(ipu_res[ptr[i - 1]:ptr[i]].tolist())
        ref_res_slice = set(ref_res[ptr[i - 1]:ptr[i]].tolist())
        assert ipu_res_slice == ref_res_slice


@pytest.mark.parametrize('src_shape', [(29, 3)])
@pytest.mark.parametrize('ptr', [[0, 29], [0, 2, 6, 11, 28, 29]])
@pytest.mark.parametrize('ratio', [0.15, 0.7, 1.0])
def test_train(src_shape, ptr, ratio):
    src = torch.rand(src_shape)
    batch = torch.zeros(src_shape[0], dtype=torch.long)
    for i in range(1, len(ptr)):
        batch[ptr[i - 1]:ptr[i]] = i - 1

    deg = np.subtract(ptr[1:], ptr[0:-1])
    out_size = np.ceil(deg * ratio).astype(int)
    out_size = np.cumsum(out_size, 0)[-1]

    linear_ipu = Linear(out_size, out_size)
    linear_ref = deepcopy(linear_ipu)

    ipu_model = FpsTrainModel(poptorch.fps, linear_ipu)
    ipu_model = poptorch.trainingModel(ipu_model)
    ipu_res, ipu_loss = ipu_model(src, ptr, ratio, random_start=False)

    ref_model = FpsTrainModel(torch_cluster.fps, linear_ref)
    ref_res, ref_loss = ref_model(src, batch, ratio, random_start=False)

    rtol = 1e-05
    atol = 1e-06
    assert np.allclose(ipu_res.tolist(),
                       ref_res.tolist(),
                       rtol=rtol,
                       atol=atol)
    assert np.allclose(ipu_loss.tolist(),
                       ref_loss.tolist(),
                       rtol=rtol,
                       atol=atol)
