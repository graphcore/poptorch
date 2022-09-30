# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import pytest
import numpy as np
import helpers
import poptorch


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_weight_update_replicas(trace_model, process_id=0, num_processes=1):
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


class ModelWithLoss(torch.nn.Module):
    def __init__(self, W_init):
        super().__init__()
        self.W = torch.nn.Parameter(W_init)

    def forward(self, X):
        Z = X @ self.W
        return Z, poptorch.identity_loss(Z**2, reduction="mean")


@pytest.mark.ipuHardwareRequired
def test_per_replica_variables():
    # Split the weight tensor into 4, and the input data tensor into 2.
    tensor_shards = 4
    data_shards = 2

    # Set up the problem
    random = np.random.RandomState(seed=100)
    prob_X = random.normal(size=(24, 40)).astype(np.float32)
    prob_W_init = random.normal(size=(40, 56)).astype(
        np.float32) * (5 * 8)**-0.5
    prob_steps = 4

    # Run on the CPU
    X = torch.tensor(prob_X)
    W = torch.nn.Parameter(torch.tensor(prob_W_init))
    optim = torch.optim.SGD([W], lr=0.01)

    cpu_losses = []
    for _ in range(prob_steps):
        optim.zero_grad()
        v = (X @ W)**2
        loss = torch.mean(v)
        loss.backward()
        optim.step()
        cpu_losses.append(loss.detach())
    cpu_losses = np.array(cpu_losses)
    cpu_W_final = W.detach().numpy()

    # Run on 8 IPUs
    W_init = torch.tensor(
        prob_W_init.reshape(prob_W_init.shape[0], tensor_shards,
                            prob_W_init.shape[1] // tensor_shards).transpose(
                                1, 0, 2)).contiguous()
    m = ModelWithLoss(W_init)
    optim = torch.optim.SGD(m.parameters(), lr=0.01)

    pt_opts = poptorch.Options()
    pt_opts.replicationFactor(data_shards * tensor_shards, data_shards)
    pt_opts.outputMode(poptorch.OutputMode.All)
    pt_m = poptorch.trainingModel(m, optimizer=optim, options=pt_opts)
    pt_m.W.perReplica(poptorch.enums.CommGroupType.Orthogonal, data_shards,
                      poptorch.enums.VariableRetrievalMode.OnePerGroup)
    pt_losses = []
    if data_shards > 1:
        X = X.reshape(data_shards, X.shape[0] // data_shards, *X.shape[1:])
    for _ in range(prob_steps):
        _, loss = pt_m(X)
        # We divide by the number of replicas because the mean is being
        # taken only over a part of the tensor on each replica, so we need to
        # divide by the number of replicas to get the correct mean.
        pt_losses.append(
            torch.sum(loss.detach()) / (data_shards * tensor_shards))
    pt_losses = np.array(pt_losses)
    pt_W_final = m.W.detach().numpy().transpose(1, 0, 2) \
                  .reshape(prob_W_init.shape)
    np.testing.assert_allclose(cpu_losses, pt_losses, atol=1e-6)
    np.testing.assert_allclose(cpu_W_final, pt_W_final, atol=1e-6)
