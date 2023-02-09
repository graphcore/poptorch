#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import pytest
import helpers
import poptorch


def int_mean(x):
    return torch.mean(x.to(torch.float))


def int_var(x):
    return torch.var(x.to(torch.float))


# Random Number Generation Harness
# Checks that the IPU generated data with roughly the same summary
# statistics as the CPU version.
def rng_harness(rng_op, input, stat_funs, expected_dtype=torch.float):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
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
        print("  {} = poptorch {}, native {}".format(ss.__name__, ss(pop_out),
                                                     ss(native_out)),
              flush=True)
        helpers.assert_allclose(expected=ss(native_out),
                                actual=ss(pop_out),
                                atol=1e-2,
                                rtol=0.1)


# torch.rand
@pytest.mark.ipuHardwareRequired
def test_rand():
    def rng_op(x):
        return torch.rand(x.size())

    stat_funs = [torch.min, torch.max, torch.mean, torch.var]
    input = torch.empty(size=(3, 5, 100))
    rng_harness(rng_op, input, stat_funs)


# torch.distributions.Uniform
@pytest.mark.ipuHardwareRequired
def test_distributions_uniform():
    def rng_op(x):
        ud = torch.distributions.Uniform(0.0, 10.0)
        return ud.sample(x.size())

    sample_like = torch.empty(10, 10, 1000)
    stat_funs = [torch.min, torch.max, torch.mean, torch.var]
    rng_harness(rng_op, sample_like, stat_funs)


# torch.uniform_
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("dt", [torch.float, torch.half])
def test_uniform_(dt):
    def rng_op(x):
        return x.uniform_()

    input = torch.empty(size=(3, 4, 1000), dtype=dt)
    stat_funs = [torch.min, torch.max, torch.mean, torch.var]
    rng_harness(rng_op, input, stat_funs, expected_dtype=dt)


# torch.normal
@pytest.mark.ipuHardwareRequired
def test_normal():
    def rng_op(x):
        return torch.normal(mean=0.0, std=1.0, size=x.size())

    input = torch.empty(6, 10, 1000)
    stat_funs = [torch.mean, torch.var]
    rng_harness(rng_op, input, stat_funs)


# torch.normal_
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("dt", [torch.float, torch.half])
def test_normal_(dt):
    def rng_op(x):
        return x.normal_(mean=1.0, std=2.0)

    input = torch.empty(size=(3, 5, 1000), dtype=dt)
    stat_funs = [torch.mean, torch.var]
    rng_harness(rng_op, input, stat_funs, expected_dtype=dt)


# torch.normal with buffers and params
@pytest.mark.ipuHardwareRequired
def test_normal_buffers():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.Tensor([1.0, 2.0, 3.0]))
            self.register_parameter(
                "std", torch.nn.Parameter(torch.Tensor([0.5, 1.0, 1.5])))

        def forward(self, x):
            torch.manual_seed(42)
            return torch.normal(self.mean, 0.5) + torch.normal(1.0,
                                                               self.std) + x

    model = Model()

    # Run on IPU and check that the result has the correct type
    opts = poptorch.Options().randomSeed(8)
    pop_model = poptorch.inferenceModel(model, opts)
    pop_out = pop_model(torch.tensor([0.0, 0.0, 0.0]))
    assert pop_out.dtype == torch.float

    native_out = model(torch.tensor([0.0, 0.0, 0.0]))
    assert native_out.size() == pop_out.size()


# torch.distributions.Normal
# The sample method uses torch.normal(Tensor mean, Tensor std)
@pytest.mark.ipuHardwareRequired
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
@pytest.mark.ipuHardwareRequired
def test_randn():
    def rng_op(x):
        return torch.randn(x.size())

    input = torch.empty(3, 5, 10000)
    stat_funs = [torch.mean, torch.var]
    rng_harness(rng_op, input, stat_funs)


# torch.random_
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("input", [
    torch.empty(3, 5, 10000, dtype=torch.float),
    torch.empty(3, 5, 10000, dtype=torch.int),
])
def test_random(input):
    def rng_op(x):
        return x.random_(5, 100)

    stat_funs = [torch.min, torch.max, int_mean, int_var]
    rng_harness(rng_op, input, stat_funs, input.dtype)


# torch.randint
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("dtype", [None, torch.int32, torch.half, torch.float])
def test_randint(dtype):
    def rng_op(x):
        return torch.randint(5, 100, x.size(), dtype=dtype)

    input = torch.empty(3, 5, 10000)
    stat_funs = [torch.min, torch.max, int_mean, int_var]
    rng_harness(rng_op, input, stat_funs,
                torch.int32 if dtype is None else dtype)


# torch.normal(Tensor mean, float std)
@pytest.mark.ipuHardwareRequired
def test_normal_tensor_mean():
    def rng_op(x):
        return torch.normal(mean=x, std=3.0)

    mean = torch.full(size=(10000, 2), fill_value=4.0)
    stat_funs = [torch.mean, torch.std]
    rng_harness(rng_op, mean, stat_funs)


# torch.normal(float mean, Tensor std)
@pytest.mark.ipuHardwareRequired
def test_normal_tensor_std():
    def rng_op(x):
        return torch.normal(mean=3.0, std=x)

    std = torch.full(size=(10000, 2), fill_value=9.0)
    stat_funs = [torch.mean, torch.std]
    rng_harness(rng_op, std, stat_funs)


# torch.bernoulli - test with both float and half types
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("t", [torch.float, torch.half])
def test_bernoulli(t):
    prob = torch.full(size=(3, 5, 100), dtype=t, fill_value=0.5)
    stat_funs = [torch.min, torch.max, torch.mean]
    rng_harness(torch.bernoulli, prob, stat_funs, expected_dtype=t)


# torch.bernoulli - check expected output for probability limits.
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("p", [0.0, 1.0])
def test_bernoulli_limits(p):
    prob = torch.full(size=(3, 5, 1000), fill_value=p)
    func = lambda x: torch.all(x == p)
    func.__name__ = f"torch.all(x == {p})"
    rng_harness(torch.bernoulli, prob, [func])


# torch.bernoulli_
@pytest.mark.ipuHardwareRequired
def test_bernoulli_():
    def rng_op(x):
        return x.bernoulli_(p=0.3)

    input = torch.empty(3, 5, 100)
    stat_funs = [torch.min, torch.max, torch.mean]
    rng_harness(rng_op, input, stat_funs)


# torch.distributions.Bernoulli
@pytest.mark.ipuHardwareRequired
def test_distributions_bernoulli():
    def rng_op(x):
        bd = torch.distributions.Bernoulli(0.5)
        return bd.sample(x.size())

    input = torch.empty(10, 10, 1000)
    stat_funs = [torch.min, torch.max, torch.mean]
    rng_harness(rng_op, input, stat_funs)


# torch.exponential_
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("lambd", [0.5, 1.0])
def test_exponential_(lambd):
    def rng_op(x):
        return x.exponential_(lambd=lambd)

    input = torch.empty(3, 5, 100)
    stat_funs = [torch.mean]
    rng_harness(rng_op, input, stat_funs)


# torch.distributions.Exponential
@pytest.mark.ipuHardwareRequired
def test_distributions_exponential():
    def rng_op(x):
        bd = torch.distributions.Exponential(0.5)
        return bd.sample(x.size())

    input = torch.empty(10, 10, 1000)
    stat_funs = [torch.mean]
    rng_harness(rng_op, input, stat_funs)


@pytest.mark.ipuHardwareRequired
def test_randperm():
    def rng_op(x):
        return torch.randperm(x.size(dim=0)) + 0

    input = torch.arange(100)
    stat_funs = [torch.numel]
    rng_harness(rng_op, input, stat_funs, torch.int32)


@pytest.mark.ipuHardwareRequired
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
