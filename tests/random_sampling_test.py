#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import pytest
import poptorch
import helpers


# Random Number Generation Harness
# Checks that the IPU generated data with roughly the same summary
# statistics as the CPU version.
def rng_harness(trace_model,
                rng_op,
                input,
                stat_funs,
                expected_dtype=torch.float):
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
    opts.Jit.traceModel(trace_model)
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


# torch.rand
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_rand(trace_model):
    def rng_op(x):
        return torch.rand(x.size())

    stat_funs = [torch.min, torch.max, torch.mean, torch.var]
    input = torch.empty(size=(3, 5, 100))
    rng_harness(trace_model, rng_op, input, stat_funs)


# torch.distributions.Uniform
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_distributions_uniform(trace_model):
    def rng_op(x):
        ud = torch.distributions.Uniform(0.0, 10.0)
        return ud.sample(x.size())

    sample_like = torch.empty(10, 10, 1000)
    stat_funs = [torch.min, torch.max, torch.mean, torch.var]
    rng_harness(trace_model, rng_op, sample_like, stat_funs)


# torch.uniform_
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("dt", [torch.float, torch.half])
@pytest.mark.parametrize("trace_model", [True, False])
def test_uniform_(dt, trace_model):
    def rng_op(x):
        return x.uniform_()

    input = torch.empty(size=(3, 4, 1000), dtype=dt)
    stat_funs = [torch.min, torch.max, torch.mean, torch.var]
    rng_harness(trace_model, rng_op, input, stat_funs, expected_dtype=dt)


# torch.normal
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_normal(trace_model):
    def rng_op(x):
        return torch.normal(mean=0.0, std=1.0, size=x.size())

    input = torch.empty(6, 10, 1000)
    stat_funs = [torch.mean, torch.var]
    rng_harness(trace_model, rng_op, input, stat_funs)


# torch.normal_
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("dt", [torch.float, torch.half])
@pytest.mark.parametrize("trace_model", [True, False])
def test_normal_(dt, trace_model):
    def rng_op(x):
        return x.normal_(mean=1.0, std=2.0)

    input = torch.empty(size=(3, 5, 1000), dtype=dt)
    stat_funs = [torch.mean, torch.var]
    rng_harness(trace_model, rng_op, input, stat_funs, expected_dtype=dt)


# torch.distributions.Normal
# The sample method uses torch.normal(Tensor mean, Tensor std)
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_distributions_normal(trace_model):
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
    rng_harness(trace_model, rng_op, input, stat_funs)


# torch.randn
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_randn(trace_model):
    def rng_op(x):
        return torch.randn(x.size())

    input = torch.empty(3, 5, 10000)
    stat_funs = [torch.mean, torch.var]
    rng_harness(trace_model, rng_op, input, stat_funs)


# torch.normal(Tensor mean, float std)
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_normal_tensor_mean(trace_model):
    def rng_op(x):
        return torch.normal(mean=x, std=3.0)

    mean = torch.full(size=(10000, 2), fill_value=4.0)
    stat_funs = [torch.mean, torch.std]
    rng_harness(trace_model, rng_op, mean, stat_funs)


# torch.normal(float mean, Tensor std)
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_normal_tensor_std(trace_model):
    def rng_op(x):
        return torch.normal(mean=3.0, std=x)

    std = torch.full(size=(10000, 2), fill_value=9.0)
    stat_funs = [torch.mean, torch.std]
    rng_harness(trace_model, rng_op, std, stat_funs)


# torch.bernoulli - test with both float and half types
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("t", [torch.float, torch.half])
@pytest.mark.parametrize("trace_model", [True, False])
def test_bernoulli(t, trace_model):
    prob = torch.full(size=(3, 5, 100), dtype=t, fill_value=0.5)
    stat_funs = [torch.min, torch.max, torch.mean]
    rng_harness(trace_model,
                torch.bernoulli,
                prob,
                stat_funs,
                expected_dtype=t)


# torch.bernoulli - check expected output for probability limits.
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("p", [0.0, 1.0])
@pytest.mark.parametrize("trace_model", [True, False])
def test_bernoulli_limits(p, trace_model):
    prob = torch.full(size=(3, 5, 1000), fill_value=p)
    func = lambda x: torch.all(x == p)
    func.__name__ = f"torch.all(x == {p})"
    rng_harness(trace_model, torch.bernoulli, prob, [func])


# torch.bernoulli_
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_bernoulli_(trace_model):
    def rng_op(x):
        return x.bernoulli_(p=0.5)

    input = torch.empty(3, 5, 100)
    stat_funs = [torch.min, torch.max, torch.mean]
    rng_harness(trace_model, rng_op, input, stat_funs)


# torch.distributions.Bernoulli
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_distributions_bernoulli(trace_model):
    def rng_op(x):
        bd = torch.distributions.Bernoulli(0.5)
        return bd.sample(x.size())

    input = torch.empty(10, 10, 1000)
    stat_funs = [torch.min, torch.max, torch.mean]
    rng_harness(trace_model, rng_op, input, stat_funs)


# torch.exponential_
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("lambd", [0.5, 1.0])
@pytest.mark.parametrize("trace_model", [True, False])
def test_exponential_(trace_model, lambd):
    def rng_op(x):
        return x.exponential_(lambd=lambd)

    input = torch.empty(3, 5, 100)
    stat_funs = [torch.mean]
    rng_harness(trace_model, rng_op, input, stat_funs)


# torch.distributions.Exponential
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_distributions_exponential(trace_model):
    def rng_op(x):
        bd = torch.distributions.Exponential(0.5)
        return bd.sample(x.size())

    input = torch.empty(10, 10, 1000)
    stat_funs = [torch.mean]
    rng_harness(trace_model, rng_op, input, stat_funs)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_randperm(trace_model):
    def rng_op(x):
        return torch.randperm(x.item()) + 0

    input = torch.tensor([100], dtype=torch.int)
    stat_funs = [torch.numel]
    rng_harness(trace_model, rng_op, input, stat_funs, torch.int32)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_random_seed_repeatability(trace_model):
    class Model(torch.nn.Module):
        def forward(self, x):
            x = x + 0  # Ensure input is not modified in place
            return x.normal_()

    # Run the model once with a random seed
    model = Model()
    opts = poptorch.Options().randomSeed(42)
    opts.Jit.traceModel(trace_model)
    first_model = poptorch.inferenceModel(model, opts)
    first_run = first_model(torch.empty((2, 2)))

    # Second run with the same seed should produce identical results
    second_model = poptorch.inferenceModel(model, opts)
    second_run = second_model(torch.empty((2, 2)))
    helpers.assert_allequal(expected=first_run, actual=second_run)
