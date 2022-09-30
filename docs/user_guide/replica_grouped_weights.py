# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy
import torch
import poptorch


# groupedweights_start
class ModelWithLoss(torch.nn.Module):
    def __init__(self, W_init):
        super().__init__()
        self.W = torch.nn.Parameter(W_init)

    def forward(self, X):
        Z = X @ self.W
        return Z, poptorch.identity_loss(Z**2, reduction="mean")


# Split the weight tensor into 4, and the input data tensor into 2.
tensor_shards = 4
data_shards = 2

# Set up the problem
random = numpy.random.RandomState(seed=100)
prob_X = random.normal(size=(24, 40)).astype(numpy.float32)
prob_W_init = random.normal(size=(40, 56)).astype(
    numpy.float32) * (5 * 8)**-0.5
prob_steps = 4

X = torch.tensor(prob_X)

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
    pt_losses.append(torch.sum(loss.detach()) / (data_shards * tensor_shards))
pt_losses = numpy.array(pt_losses)
pt_W_final = m.W.detach().numpy().transpose(1, 0, 2) \
              .reshape(prob_W_init.shape)

# groupedweights_end
