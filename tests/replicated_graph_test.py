# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import multiprocessing
import os
import numpy as np
import torch
import poptorch
import helpers
import pytest
from poptorch.distributed import VirtualIpuManager as vipu

partition_name = "poptorch_tests"


def run_test(process_id=0, num_processes=1):
    localReplicationFactor = 2

    opts = poptorch.Options()
    opts.replicationFactor(localReplicationFactor)
    opts.Distributed.configureProcessId(process_id, num_processes)
    opts.Distributed.IPUoFConfigFiles(f"~/.ipuof.conf.d/{partition_name}_*")

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

        def forward(self, input):
            # Perform the GEMM operation
            x = alpha * self.matmul(input, self.B) + beta * self.C
            return x

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
            o = module(a)
            outputs = outputs + (o, )
            loss = torch.nn.L1Loss(reduction="mean")
            target = torch.zeros(o.size())
            output = loss(o, target)
            output.backward()

        # Update the weights
        optimizer.step()

        # Only keep the output slice corresponding to this process
        outputs = outputs[opts.Distributed.processId *
                          localReplicationFactor:][:localReplicationFactor]
        return [torch.cat(outputs), module.B.data, module.C.data]

    model = Model()
    poptorch_model = helpers.trainingModelWithLoss(
        model,
        options=opts,
        loss=torch.nn.L1Loss(reduction="mean"),
        optimizer=torch.optim.SGD(model.parameters(),
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
        torch.testing.assert_allclose(out[idx], ref, rtol=1e-03, atol=1e-03)


def configurePartition(num_ipus, num_gcds, num_sync_replicas):
    partitions = vipu.listPartitions()
    print(str({n: str(p) for n, p in partitions.items()}))
    if partition_name in partitions:
        vipu.deletePartition(partition_name)

    vipu.createPartition(
        partition_name,
        poptorch.distributed.Partition(num_ipus, num_gcds, num_sync_replicas))
    vipu.resetPartition(partition_name)


# Note: poptorch.ipuHardwareIsAvailable() cannot be used before the Options
# set IPUOF_CONFIG_PATH.
@pytest.mark.skipif(int(os.environ.get("POPTORCH_IPU_MODEL", "0")) != 0,
                    reason="Hardware IPU needed")
@pytest.mark.skipif(int(os.environ.get("POPTORCH_SMALL_IPU_MODEL", "0")) != 0,
                    reason="Hardware IPU needed")
def test_weight_update_replicas():

    # We always need to create a partition if we're on POD.
    if vipu.isAvailable():
        configurePartition(2, 1, 2)
    run_test()


# Note: poptorch.ipuHardwareIsAvailable() cannot be used before the Options
# set IPUOF_CONFIG_PATH.
@pytest.mark.skipif(int(os.environ.get("POPTORCH_IPU_MODEL", "0")) != 0,
                    reason="Hardware IPU needed")
@pytest.mark.skipif(int(os.environ.get("POPTORCH_SMALL_IPU_MODEL", "0")) != 0,
                    reason="Hardware IPU needed")
@pytest.mark.skipif(not vipu.isAvailable(),
                    reason="vipu-cli and 8 IPUs needed")
def test_weight_update_distributed():

    # To avoid: "RuntimeError: Unable to handle autograd's threading in
    # combination with fork-based multiprocessing.
    # See https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork"
    ctx = multiprocessing.get_context("spawn")
    # Create a partition equivalent to
    # vipu-cli create partition poptorch_tests --size 4 --num-gcds 2 --gcd-sync-replicas 4
    num_gcds = 2
    configurePartition(4, num_gcds, 4)

    processes = []
    for i in range(num_gcds):
        p = ctx.Process(target=run_test, args=(i, num_gcds))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    assert all(p.exitcode == 0 for p in processes)
