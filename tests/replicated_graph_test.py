# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import pytest
import numpy as np
import helpers
import poptorch


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_weight_update_replicas(trace_model, process_id=0, num_processes=1):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): Could not find loss tensor '' in main graph tensors"
        )
    localReplicationFactor = 2

    opts = poptorch.Options()
    opts.replicationFactor(localReplicationFactor)
    opts.Distributed.configureProcessId(process_id, num_processes)
    opts.Jit.traceModel(trace_model)

    replicationFactor = localReplicationFactor * opts.Distributed.numProcesses

    np.random.seed(42)

    A = np.random.rand(2, 4).astype(np.float32)
    B = np.ones((4, 6)).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)

    alpha = np.random.random(1).astype(np.float32)[0]
    beta = np.random.random(1).astype(np.float32)[0]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.b = torch.tensor(B, requires_grad=True)
            self.c = torch.tensor(C, requires_grad=True)

            # Create the weight tensors for pytorch
            self.B = torch.nn.Parameter(self.b, requires_grad=True)
            self.C = torch.nn.Parameter(self.c, requires_grad=True)

            self.matmul = torch.matmul

            self.loss = torch.nn.L1Loss(reduction="mean")

        def forward(self, input, target):
            # Perform the GEMM operation
            x = alpha * self.matmul(input, self.B) + beta * self.C
            loss = self.loss(x, target)
            return x, loss

    def reference():
        module = Model()
        module.train()

        optimizer = torch.optim.SGD(module.parameters(),
                                    lr=0.01,
                                    weight_decay=0.0,
                                    momentum=0.0)

        a = torch.tensor(A, requires_grad=True)
        optimizer.zero_grad()

        outputs = ()

        # graph with gradient accumlation i.e. only update the weights after x passes
        for _ in range(replicationFactor):
            target = torch.zeros(C.shape)
            out, loss = module(a, target)
            outputs = outputs + (out, )
            loss.backward()

        # Update the weights
        optimizer.step()

        # Only keep the output slice corresponding to this process
        outputs = outputs[opts.Distributed.processId *
                          localReplicationFactor:][:localReplicationFactor]
        return [torch.cat(outputs), module.B.data, module.C.data]

    model = Model()
    poptorch_model = poptorch.trainingModel(model,
                                            options=opts,
                                            optimizer=torch.optim.SGD(
                                                model.parameters(),
                                                lr=0.01,
                                                weight_decay=0.0,
                                                momentum=0.0))

    ref_out = reference()
    ipu_A = np.concatenate([A for _ in range(localReplicationFactor)])

    target = torch.zeros(2 * localReplicationFactor, 6)
    output, _ = poptorch_model(torch.tensor(ipu_A, requires_grad=True), target)
    out = [output, model.B.data, model.C.data]
    for idx, ref in enumerate(ref_out):
        print("Validating output %d" % idx)
        helpers.assert_allclose(actual=out[idx],
                                expected=ref,
                                rtol=1e-03,
                                atol=1e-03)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_too_many_ipus(trace_model):
    localReplicationFactor = 128

    opts = poptorch.Options()
    opts.replicationFactor(localReplicationFactor)
    opts.Jit.traceModel(trace_model)

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.layer = torch.nn.Linear(128, 4)
            self.loss = torch.nn.L1Loss(reduction="mean")

        def forward(self, input, target):
            out = self.layer(input)
            loss = self.loss(out, target)
            return out, loss

    model = Model()

    poptorch_model = poptorch.trainingModel(model,
                                            options=opts,
                                            optimizer=torch.optim.SGD(
                                                model.parameters(),
                                                lr=0.01,
                                                weight_decay=0.0,
                                                momentum=0.0))

    np.random.seed(42)
    input = np.random.rand(512, 128).astype(np.float32)
    labels = np.ones((128, 4)).astype(np.float32)

    with pytest.raises(
            poptorch.Error,
            match=r"Too many IPUs requested \(128\)\. Experiments that need .*"
    ):
        poptorch_model(torch.tensor(input, requires_grad=True),
                       torch.tensor(labels))
