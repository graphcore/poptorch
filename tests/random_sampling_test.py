#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import pytest
import poptorch
import helpers


# Random Number Generation Harness
# Checks that the IPU generated data with roughly the same summary
# statistics as the CPU version.
def rng_harness(rng_op, input, stat_funs, expected_dtype=torch.float):
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.rng_op = rng_op

        def forward(self, x):
            torch.manual_seed(42)
            x = x + 0  # Ensure input is not modified in place
            return self.rng_op(x)

    model = Model()

    # Run on IPU and check that the result has the correct type
    opts = poptorch.Options().randomSeed(8)
    pop_model = poptorch.inferenceModel(model, opts)
    pop_out = pop_model(input)
    assert pop_out.dtype == expected_dtype

    if expected_dtype is torch.half:
        # Promote CPU model and input
        model = model.float()
        input = input.float()
        # promote IPU result to allow summary stat comparison
        pop_out = pop_out.float()

    native_out = model(input)
    assert native_out.size() == pop_out.size()

    # PRNG depends on HW implementation so we just check
    # that the distribution statistics are consistent
    print("Checking summary statistics for generated random numbers:")
    for ss in stat_funs:
        print("  {} = {}".format(ss.__name__, ss(pop_out)))
        helpers.assert_allclose(expected=ss(native_out),
                                actual=ss(pop_out),
                                atol=1e-2,
                                rtol=0.1)


def rng_requirements(func):
    # Filter the following expected warnings for RNG tests
    warnings = [
        "ignore:Trace had nondeterministic nodes",
        "ignore:Output nr 1. of the traced function does not match",
        "ignore:torch.tensor results are registered as constants in the trace"
    ]

    markers = [pytest.mark.filterwarnings(w) for w in warnings]

    # PRNG requires IPU hardware so skip if not available
    skip = pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                              reason="Hardware IPU needed")
    markers.append(skip)

    for m in markers:
        func = m(func)

    return func


# torch.rand
@rng_requirements
def test_rand():
    def rng_op(x):
        return torch.rand(x.size())

    stat_funs = [torch.min, torch.max, torch.mean, torch.var]
    input = torch.empty(size=(3, 5, 100))
    rng_harness(rng_op, input, stat_funs)


# torch.distributions.Uniform
@rng_requirements
def test_distributions_uniform():
    def rng_op(x):
        ud = torch.distributions.Uniform(0.0, 10.0)
        return ud.sample(x.size())

    sample_like = torch.empty(10, 10, 1000)
    stat_funs = [torch.min, torch.max, torch.mean, torch.var]
    rng_harness(rng_op, sample_like, stat_funs)


# torch.uniform_
@rng_requirements
@pytest.mark.parametrize("dt", [torch.float, torch.half])
def test_uniform_(dt):
    def rng_op(x):
        return x.uniform_()

    input = torch.empty(size=(3, 4, 1000), dtype=dt)
    stat_funs = [torch.min, torch.max, torch.mean, torch.var]
    rng_harness(rng_op, input, stat_funs, expected_dtype=dt)


# torch.normal
@rng_requirements
def test_normal():
    def rng_op(x):
        return torch.normal(mean=0.0, std=1.0, size=x.size())

    input = torch.empty(6, 10, 1000)
    stat_funs = [torch.mean, torch.var]
    rng_harness(rng_op, input, stat_funs)


# torch.normal_
@rng_requirements
@pytest.mark.parametrize("dt", [torch.float, torch.half])
def test_normal_(dt):
    def rng_op(x):
        return x.normal_(mean=1.0, std=2.0)

    input = torch.empty(size=(3, 5, 1000), dtype=dt)
    stat_funs = [torch.mean, torch.var]
    rng_harness(rng_op, input, stat_funs, expected_dtype=dt)


# torch.distributions.Normal
# The sample method uses torch.normal(Tensor mean, Tensor std)
@rng_requirements
def test_distributions_normal():
    def rng_op(x):
        h = torch.tensor([234.0, 100.0])
        nd = torch.distributions.Normal(loc=h, scale=torch.sqrt(h))
        return nd.sample(x.size())

    input = torch.empty(10000, 5)
    mean = lambda x: torch.mean(x, dim=[0, 1])
    mean.__name__ = "torch.mean(x, dim=[0, 1])"

    std = lambda x: torch.std(x, dim=[0, 1])
    std.__name__ = "torch.std(x, dim=[0, 1])"

    stat_funs = [mean, std]
    rng_harness(rng_op, input, stat_funs)


# torch.randn
@rng_requirements
def test_randn():
    def rng_op(x):
        return torch.randn(x.size())

    input = torch.empty(3, 5, 10000)
    stat_funs = [torch.mean, torch.var]
    rng_harness(rng_op, input, stat_funs)


# torch.normal(Tensor mean, float std)
@rng_requirements
def test_normal_tensor_mean():
    def rng_op(x):
        return torch.normal(mean=x, std=3.0)

    mean = torch.full(size=(10000, 2), fill_value=4.0)
    stat_funs = [torch.mean, torch.std]
    rng_harness(rng_op, mean, stat_funs)


# torch.normal(float mean, Tensor std)
@rng_requirements
def test_normal_tensor_std():
    def rng_op(x):
        return torch.normal(mean=3.0, std=x)

    std = torch.full(size=(10000, 2), fill_value=9.0)
    stat_funs = [torch.mean, torch.std]
    rng_harness(rng_op, std, stat_funs)


# torch.bernoulli - test with both float and half types
@rng_requirements
@pytest.mark.parametrize("t", [torch.float, torch.half])
def test_bernoulli(t):
    prob = torch.full(size=(3, 5, 100), dtype=t, fill_value=0.5)
    stat_funs = [torch.min, torch.max, torch.mean]
    rng_harness(torch.bernoulli, prob, stat_funs, expected_dtype=t)


# torch.bernoulli - check expected output for probability limits.
@rng_requirements
@pytest.mark.parametrize("p", [0.0, 1.0])
def test_bernoulli_limits(p):
    prob = torch.full(size=(3, 5, 1000), fill_value=p)
    func = lambda x: torch.all(x == p)
    func.__name__ = f"torch.all(x == {p})"
    rng_harness(torch.bernoulli, prob, [func])


# torch.bernoulli_
@rng_requirements
def test_bernoulli_():
    def rng_op(x):
        return x.bernoulli_(p=0.5)

    input = torch.empty(3, 5, 100)
    stat_funs = [torch.min, torch.max, torch.mean]
    rng_harness(rng_op, input, stat_funs)


# torch.distributions.Bernoulli
@rng_requirements
def test_distributions_bernoulli():
    def rng_op(x):
        bd = torch.distributions.Bernoulli(0.5)
        return bd.sample(x.size())

    input = torch.empty(10, 10, 1000)
    stat_funs = [torch.min, torch.max, torch.mean]
    rng_harness(rng_op, input, stat_funs)


@rng_requirements
def test_random_seed_repeatability():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = x + 0  # Ensure input is not modified in place
            return x.normal_()

    # Run the model once with a random seed
    model = Model()
    opts = poptorch.Options().randomSeed(42)
    first_model = poptorch.inferenceModel(model, opts)
    first_run = first_model(torch.empty((2, 2)))

    # Second run with the same seed should produce identical results
    second_model = poptorch.inferenceModel(model, opts)
    second_run = second_model(torch.empty((2, 2)))
    helpers.assert_allequal(expected=first_run, actual=second_run)
