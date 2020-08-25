#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import pytest
import poptorch


# Random Number Generation Harness
# Checks that the IPU generated data with roughly the same summary statistics as the CPU version
def rng_harness(rng_op, stat_funs):
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.rng_op = rng_op

        def forward(self):
            return self.rng_op()

    model = Model()

    # Run on CPU
    native_out = model()

    # Run on IPU
    pop_model = poptorch.inferenceModel(model)
    pop_out = pop_model()

    assert native_out.size() == pop_out.size()

    # PRNG depends on HW implementation so we just check
    # that the distribution statistics are consistent
    print("Checking summary statistics for generated random numbers:")
    for ss in stat_funs:
        print("  {} = {}".format(ss.__name__, ss(pop_out)))
        torch.testing.assert_allclose(ss(native_out),
                                      ss(pop_out),
                                      atol=1e-2,
                                      rtol=0.1)


# torch.rand
# Filter the following expected warnings
@pytest.mark.filterwarnings("ignore:Trace had nondeterministic nodes")
@pytest.mark.filterwarnings(
    "ignore:Output nr 1. of the traced function does not match")
def test_rand():
    def rng_op():
        torch.manual_seed(42)
        return torch.rand(3, 5, 100)

    stat_funs = [torch.min, torch.max, torch.mean, torch.var]

    rng_harness(rng_op, stat_funs)


# torch.distributions.uniform.Uniform
# Filter the following expected warnings
@pytest.mark.filterwarnings("ignore:Trace had nondeterministic nodes")
@pytest.mark.filterwarnings(
    "ignore:Output nr 1. of the traced function does not match")
@pytest.mark.filterwarnings(
    "ignore:torch.Tensor results are registered as constants in the trace")
def test_distributions_uniform():
    def rng_op():
        torch.manual_seed(42)
        ud = torch.distributions.uniform.Uniform(0.0, 10.0)
        return ud.sample((10, 10, 1000))

    stat_funs = [torch.min, torch.max, torch.mean, torch.var]

    rng_harness(rng_op, stat_funs)


# torch.uniform_
# Filter the following expected warnings
@pytest.mark.filterwarnings("ignore:Trace had nondeterministic nodes")
@pytest.mark.filterwarnings(
    "ignore:Output nr 1. of the traced function does not match")
def test_uniform_():
    def rng_op():
        torch.manual_seed(42)
        return torch.empty((3, 4, 1000)).uniform_()

    stat_funs = [torch.min, torch.max, torch.mean, torch.var]

    rng_harness(rng_op, stat_funs)


# torch.normal
# Filter the following expected warnings
@pytest.mark.filterwarnings("ignore:Trace had nondeterministic nodes")
@pytest.mark.filterwarnings(
    "ignore:Output nr 1. of the traced function does not match")
def test_normal():
    def rng_op():
        torch.manual_seed(42)
        return torch.normal(mean=0.0, std=1.0, size=(6, 10, 1000))

    stat_funs = [torch.mean, torch.var]

    rng_harness(rng_op, stat_funs)


# torch.normal_
# Filter the following expected warnings
@pytest.mark.filterwarnings("ignore:Trace had nondeterministic nodes")
@pytest.mark.filterwarnings(
    "ignore:Output nr 1. of the traced function does not match")
def test_normal_():
    def rng_op():
        torch.manual_seed(42)
        return torch.empty(3, 5, 1000).normal_(mean=1.0, std=2.0)

    stat_funs = [torch.mean, torch.var]

    rng_harness(rng_op, stat_funs)


# torch.distributions.Normal
# The sample method uses torch.normal(Tensor mean, Tensor std)
# Filter the following expected warnings
@pytest.mark.filterwarnings("ignore:Trace had nondeterministic nodes")
@pytest.mark.filterwarnings(
    "ignore:Output nr 1. of the traced function does not match")
@pytest.mark.filterwarnings(
    "ignore:torch.Tensor results are registered as constants in the trace")
def test_distributions_normal():
    def rng_op():
        torch.manual_seed(42)
        h = torch.tensor([234.0, 100.0])
        nd = torch.distributions.Normal(loc=h, scale=torch.sqrt(h))
        return nd.sample((10000, 5))

    # Generates (10000, 5, 2) tensor
    mean = lambda x: torch.mean(x, dim=[0, 1])
    mean.__name__ = "torch.mean(x, dim=[0, 1])"

    std = lambda x: torch.std(x, dim=[0, 1])
    std.__name__ = "torch.std(x, dim=[0, 1])"

    stat_funs = [mean, std]

    rng_harness(rng_op, stat_funs)


# torch.randn
# Filter the following expected warnings
@pytest.mark.filterwarnings("ignore:Trace had nondeterministic nodes")
@pytest.mark.filterwarnings(
    "ignore:Output nr 1. of the traced function does not match")
def test_randn():
    def rng_op():
        torch.manual_seed(42)
        return torch.randn(3, 5, 10000)

    stat_funs = [torch.mean, torch.var]

    rng_harness(rng_op, stat_funs)


# torch.normal(Tensor mean, float std)
# Filter the following expected warnings
@pytest.mark.filterwarnings("ignore:Trace had nondeterministic nodes")
@pytest.mark.filterwarnings(
    "ignore:Output nr 1. of the traced function does not match")
@pytest.mark.filterwarnings(
    "ignore:torch.Tensor results are registered as constants in the trace")
def test_normal_tensor_mean():
    def rng_op():
        torch.manual_seed(42)
        mean = torch.full(size=(10000, 2), fill_value=4.0)
        std = 3.0
        return torch.normal(mean=mean, std=std)

    stat_funs = [torch.mean, torch.std]

    rng_harness(rng_op, stat_funs)


# torch.normal(float mean, Tensor std)
# Filter the following expected warnings
@pytest.mark.filterwarnings("ignore:Trace had nondeterministic nodes")
@pytest.mark.filterwarnings(
    "ignore:Output nr 1. of the traced function does not match")
@pytest.mark.filterwarnings(
    "ignore:torch.Tensor results are registered as constants in the trace")
def test_normal_tensor_std():
    def rng_op():
        torch.manual_seed(42)
        mean = 3.0
        std = torch.full(size=(10000, 2), fill_value=9.0)
        return torch.normal(mean=mean, std=std)

    stat_funs = [torch.mean, torch.std]

    rng_harness(rng_op, stat_funs)
