# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch
from torch_geometric.nn import InstanceNorm

import helpers
from gnn.nn.nn_utils import ModelWW
import poptorch


@pytest.mark.parametrize('conf', [True, False])
def test_instance_norm(conf):
    atol = None
    rtol = None
    if conf is True:
        # These values are based on torch_nn_test.py file
        # where InstanceNorm is tested from torch package.
        atol = 1e-3
        rtol = 0.05

    nodes_list = torch.randn(5, 100, 16)

    def test_body(inputs):

        norm = InstanceNorm(16, affine=conf, track_running_stats=conf)

        cpu_model = ModelWW(norm, inputs[0][0].shape)
        ipu_model = poptorch.trainingModel(ModelWW(norm, inputs[0][0].shape))

        for x in inputs[0]:
            cpu_out = None
            ipu_out = None
            if len(inputs) > 1:
                model_inputs = [x] + inputs[1:]
                cpu_out = cpu_model(model_inputs)
                ipu_out = ipu_model(model_inputs)
            else:
                cpu_out = cpu_model([x])
                ipu_out = ipu_model([x])
            helpers.assert_allclose(actual=ipu_out[0],
                                    expected=cpu_out[0],
                                    atol=atol,
                                    rtol=rtol)

    test_body([nodes_list])

    batch = torch.zeros(100, dtype=torch.long)
    batch_size = 1
    test_body([nodes_list, batch, batch_size])

    batch[:50] = torch.ones(50, dtype=torch.long)
    batch_size = 2
    test_body([nodes_list, batch, batch_size])
