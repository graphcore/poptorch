#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import torch.nn.functional as F
import pytest
import helpers
import poptorch

# Linears
# torch.nn.Identity, torch.nn.Linear, torch.nn.Bilinear,

# Dropouts
# torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d, torch.nn.AlphaDropout,

# Sparse
# torch.nn.Embedding, torch.nn.Embedding.from_pretrained, torch.nn.EmbeddingBag, torch.nn.EmbeddingBag.from_pretrained,

include_bias = [True, False]

# The inner dimensions used in testing bilinear layers
input_feature_shapes = [
    {
        "x1": (),
        "x2": ()
    },
    {  # ND feature inputs
        "x1": (4, 5),
        "x2": (4, 5)
    }
]


def op_harness(op, inputs, inference_test_fn=None):
    if inference_test_fn is None:
        inference_test_fn = lambda native_out, poptorch_out: helpers.assert_allclose(
            expected=native_out, actual=poptorch_out)

    opts = poptorch.Options().randomSeed(42)
    model = helpers.ModelWithWeights(op, inputs[0].shape)

    # Run on CPU.
    native_out, _ = model(tuple(inputs))

    # Run on IPU.
    poptorch_model = poptorch.trainingModel(model, opts)
    poptorch_out, _ = poptorch_model(tuple(inputs))

    # Inference test - check outputs
    inference_test_fn(native_out, poptorch_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


# TODO(T26403): Re-enable floating point scales once bug in Popart fixed
#@pytest.mark.parametrize("scale_factor", [5.00001, 5.12498])
@pytest.mark.parametrize("scale_factor", [2, 3.5])
@pytest.mark.parametrize("input_shape", [(1, 2, 8), (2, 2, 2, 8),
                                         (2, 3, 4, 2, 8)])
def test_upsample_nearest(scale_factor, input_shape):
    torch.manual_seed(42)
    op = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
    x = torch.randn(*input_shape)
    op_harness(op, [x])


def test_downsample_nearest():
    torch.manual_seed(42)
    # test case from T44610
    op = torch.nn.Upsample(scale_factor=0.435714, mode="nearest")
    x = torch.randn(1, 2, 14, 14)
    op_harness(op, [x])


# TODO(T43375): replace scale factor 5 with 3.5
@pytest.mark.parametrize("scale_factor", [2, 5])
@pytest.mark.parametrize("input_shape", [(1, 2, 3, 4), (2, 2, 2, 8)])
def test_upsample_bilinear_factor(scale_factor, input_shape):
    torch.manual_seed(42)
    op = torch.nn.Upsample(scale_factor=scale_factor, mode="bilinear")
    x = torch.randn(*input_shape)
    op_harness(op, [x])


@pytest.mark.parametrize("shapes", [[(1, 2, 3, 4),
                                     (6, 8)], [(2, 2, 2, 8), (7, 28)]])
def test_upsample_bilinear_factor_shapes(shapes):
    torch.manual_seed(42)
    op = torch.nn.Upsample(size=shapes[1], mode="bilinear")
    x = torch.randn(*shapes[0])
    op_harness(op, [x])


@pytest.mark.parametrize("mode, input_shape", [("linear", (1, 2, 3)),
                                               ("bicubic", (1, 2, 3, 4)),
                                               ("trilinear", (1, 2, 3, 4, 5))])
def test_unsupported_upsample(mode, input_shape):
    torch.manual_seed(42)
    scale_factor = 2
    model = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)
    x = torch.randn(*input_shape)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    with pytest.raises(poptorch.Error, match="only 'nearest' is supported"):
        poptorch_model(x)


def test_linear():
    torch.manual_seed(42)
    model = torch.nn.Linear(20, 30)
    x = torch.randn(128, 20)

    # Run on CPU.
    native_out = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    assert native_out.size() == poptorch_out.size()
    helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.parametrize("include_bias", include_bias)
@pytest.mark.parametrize("input_feature_shapes", input_feature_shapes)
def test_bilinear(include_bias, input_feature_shapes):
    torch.manual_seed(42)
    op = torch.nn.Bilinear(20, 30, 40, bias=include_bias)
    shape1 = input_feature_shapes['x1']
    shape2 = input_feature_shapes['x2']
    x1 = torch.randn(8, *shape1, 20)
    x2 = torch.randn(8, *shape2, 30)
    op_harness(op, [x1, x2])


def test_identity():
    torch.manual_seed(42)
    op = torch.nn.Identity(20, 30, 40)
    x = torch.randn(128, 20)
    op_harness(op, [x])


dropout_ops = [torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d]


@pytest.mark.parametrize("dropout_op", dropout_ops)
def test_dropout_inference(dropout_op):
    torch.manual_seed(42)
    model = dropout_op()
    model.eval()

    x = torch.randn(128, 20)

    # Run on CPU.
    native_out = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    msg = f"{dropout_op.__name__} in inference session should equal identity."
    helpers.assert_allequal(expected=native_out, actual=poptorch_out, msg=msg)


@pytest.mark.parametrize("dropout_op", dropout_ops)
def test_dropout_eval_during_training(dropout_op):
    torch.manual_seed(42)
    dropout = dropout_op()
    dropout.eval()

    x = torch.randn(128, 20)

    # Create a model consisting of a single dropout operation
    # with a dummy parameter for the optimizer
    dropout.register_parameter('param', torch.nn.Parameter(torch.empty(10)))
    native_out = dropout(x)

    # Create a poptorch training model with a fixed random seed for deterministic runs
    # Note that the loss is irrelevant and ignored.
    opts = poptorch.Options().randomSeed(8)
    poptorch_model = helpers.trainingModelWithLoss(dropout,
                                                   loss=torch.nn.L1Loss(),
                                                   options=opts)
    dummy_label = torch.zeros_like(x)
    poptorch_out, _ = poptorch_model(x, dummy_label)

    assert native_out.size() == poptorch_out.size()
    msg = f"{dropout_op.__name__} should equal identity."
    helpers.assert_allequal(expected=x, actual=poptorch_out, msg=msg)


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_dropout_training():
    drop_ratio = 0.8
    dropout_op = torch.nn.Dropout(drop_ratio)

    # Input size needs to be large enough for convergence to expected dropout ratio
    sz = [100, 4, 3]
    x = torch.ones(sz, dtype=torch.float)

    def check_ratio(_, poptorch_out):
        # Instead we test that poptorch converge to the expected dropout ratio
        actual_ratio = x[poptorch_out == 0].sum() / x.numel()
        helpers.assert_allclose(actual=actual_ratio,
                                expected=drop_ratio,
                                rtol=0.01,
                                atol=0.01)

    op_harness(dropout_op, [x], check_ratio)


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_dropout2d_training():
    drop_ratio = 0.8
    dropout_op = torch.nn.Dropout2d(drop_ratio)

    # Input size needs to be large enough for convergence to expected dropout ratio
    N = 30
    C = 30
    num_channels = torch.as_tensor(N * C, dtype=torch.float)
    sz = [N, C, 2, 2]
    x = torch.ones(sz, dtype=torch.float)

    def check_ratio(_, poptorch_out):
        channel_mask = (poptorch_out == 0).all(-1).all(-1)
        actual_ratio = channel_mask.sum() / num_channels
        helpers.assert_allclose(actual=actual_ratio,
                                expected=drop_ratio,
                                rtol=0.01,
                                atol=0.01)

    op_harness(dropout_op, [x], check_ratio)


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_dropout3d_training():
    drop_ratio = 0.6
    dropout_op = torch.nn.Dropout3d(drop_ratio)

    # Input size needs to be large enough for convergence to expected dropout ratio
    N = 30
    C = 30
    num_channels = torch.as_tensor(N * C, dtype=torch.float)
    sz = [N, C, 2, 2, 1]
    x = torch.ones(sz, dtype=torch.float)

    def check_ratio(_, poptorch_out):
        channel_mask = (poptorch_out == 0).all(-1).all(-1).all(-1)
        actual_ratio = channel_mask.sum() / num_channels
        helpers.assert_allclose(actual=actual_ratio,
                                expected=drop_ratio,
                                rtol=0.01,
                                atol=0.01)

    op_harness(dropout_op, [x], check_ratio)


def test_embedding():
    model = torch.nn.Embedding(10, 3)
    x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])

    # Run on CPU.
    native_out = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    assert native_out.size() == poptorch_out.size()
    helpers.assert_allequal(expected=native_out, actual=poptorch_out)


@pytest.mark.parametrize("mode", ["max", "mean", "sum"])
def test_embedding_bag(mode):
    torch.manual_seed(0)
    model = torch.nn.EmbeddingBag(10, 3, mode=mode)
    x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    cpu_out = model(x)
    pop_model = poptorch.inferenceModel(model)
    pop_out = pop_model(x)
    helpers.assert_allclose(actual=pop_out, expected=cpu_out)


def test_embedding_bag_per_sample_weights():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            # per_sample_weights are only supported for mode="sum"
            self.embedding_bag = torch.nn.EmbeddingBag(10, 3, mode="sum")
            self.per_sample_weights = torch.randn(2, 4)

        def forward(self, x):
            return self.embedding_bag(
                x, per_sample_weights=self.per_sample_weights)

    torch.manual_seed(0)
    model = Model()
    x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    cpu_out = model(x)
    pop_model = poptorch.inferenceModel(model)
    pop_out = pop_model(x)
    helpers.assert_allclose(actual=pop_out, expected=cpu_out)


@pytest.mark.parametrize("mode", ["max", "mean", "sum"])
def test_embedding_bag_include_last_offset(mode):
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.weight = torch.nn.Parameter(torch.Tensor(10, 3))
            torch.nn.init.normal_(self.weight)

        def forward(self, x):
            offsets = torch.arange(0, x.numel(), x.size(1))
            offsets = torch.cat((offsets, torch.tensor([x.numel()])))
            x = x.reshape(-1)
            return F.embedding_bag(x,
                                   self.weight,
                                   offsets=offsets,
                                   include_last_offset=True,
                                   mode=mode)

    torch.manual_seed(0)
    model = Model()
    x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    cpu_out = model(x)
    pop_model = poptorch.inferenceModel(model)
    pop_out = pop_model(x)
    helpers.assert_allclose(actual=pop_out, expected=cpu_out)


def test_pixel_shuffle():
    torch.manual_seed(42)
    op = torch.nn.PixelShuffle(3)
    x = torch.randn(2, 18, 4, 4)
    op_harness(op, [x])
