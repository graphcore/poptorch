#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
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


# TODO(T26403): Re-enable floating point scales once bug in Popart fixed
#@pytest.mark.parametrize("scale_factor", [5.00001, 5.12498])
@pytest.mark.parametrize("scale_factor", [2, 3.5])
@pytest.mark.parametrize("input_shape", [(1, 2, 8), (2, 2, 2, 8),
                                         (2, 3, 4, 2, 8)])
def test_upsample(scale_factor, input_shape):
    mode = "nearest"  # Other modes not supported by Popart
    model = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)
    x = torch.randn(*input_shape)

    # Run on CPU.
    nativeOut = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    assert nativeOut.size() == poptorch_out.size()
    torch.testing.assert_allclose(nativeOut, poptorch_out)


@pytest.mark.parametrize("mode, input_shape", [("linear", (1, 2, 3)),
                                               ("bilinear", (1, 2, 3, 4)),
                                               ("bicubic", (1, 2, 3, 4)),
                                               ("trilinear", (1, 2, 3, 4, 5))])
def test_unsupported_upsample(mode, input_shape):
    scale_factor = 2
    model = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)
    x = torch.randn(*input_shape)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    with pytest.raises(RuntimeError, match="only 'nearest' is supported"):
        poptorch_model(x)


def test_linear():
    model = torch.nn.Linear(20, 30)
    x = torch.randn(128, 20)

    # Run on CPU.
    nativeOut = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    assert nativeOut.size() == poptorch_out.size()
    torch.testing.assert_allclose(nativeOut, poptorch_out)


@pytest.mark.parametrize("include_bias", include_bias)
@pytest.mark.parametrize("input_feature_shapes", input_feature_shapes)
def test_bilinear(include_bias, input_feature_shapes):
    model = torch.nn.Bilinear(20, 30, 40, bias=include_bias)
    shape1 = input_feature_shapes['x1']
    shape2 = input_feature_shapes['x2']
    x1 = torch.randn(8, *shape1, 20)
    x2 = torch.randn(8, *shape2, 30)

    # Run on CPU
    native_out = model(x1, x2)

    # Run on IPU
    poptorch_model = poptorch.inferenceModel(model)
    actual = poptorch_model(x1, x2)

    assert native_out.size() == actual.size()
    torch.testing.assert_allclose(native_out, actual)


def test_identity():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.op = torch.nn.Identity(20, 30, 40)

        def forward(self, x, y):
            # Make the graph compile
            return self.op(x) + y

    model = Model()

    x = torch.randn(128, 20)
    y = torch.zeros_like(x)

    # Run on CPU.
    nativeOut = model(x, y)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x, y)

    assert nativeOut.size() == poptorch_out.size()
    assert torch.equal(nativeOut, poptorch_out)


dropout_ops = [torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d]


@pytest.mark.parametrize("dropout_op", dropout_ops)
def test_dropout_inference(dropout_op):
    model = dropout_op()
    model.eval()

    torch.manual_seed(0)
    x = torch.randn(128, 20)

    # Run on CPU.
    native_out = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    msg = f"{dropout_op.__name__} in inference session should equal identity."
    torch.testing.assert_allclose(poptorch_out,
                                  native_out,
                                  rtol=0,
                                  atol=0,
                                  msg=msg)


def dropout_training_harness(dropout_op, input, check_func):
    # Create a model consisting of a single dropout operation
    # with a dummy parameter for the optimizer
    model = dropout_op
    model.register_parameter('param', torch.nn.Parameter(torch.empty(10)))
    torch.manual_seed(0)
    native_out = model(input)

    # Create a poptorch training model with a fixed random seed for deterministic runs
    # Note that the loss is irrelevant and ignored.
    opts = poptorch.Options().randomSeed(8)
    poptorch_model = helpers.trainingModelWithLoss(model,
                                                   loss=torch.nn.L1Loss(),
                                                   options=opts)
    dummy_label = torch.zeros_like(input)
    poptorch_out, _ = poptorch_model(input, dummy_label)
    assert native_out.size() == poptorch_out.size()
    check_func(poptorch_out)


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_dropout_training():
    drop_ratio = 0.8
    dropout_op = torch.nn.Dropout(drop_ratio)

    # Input size needs to be large enough for convergence to expected dropout ratio
    sz = [100, 4, 3]
    x = torch.ones(sz, dtype=torch.float)

    def check_ratio(poptorch_out):
        # Instead we test that poptorch converge to the expected dropout ratio
        actual_ratio = x[poptorch_out == 0].sum() / x.numel()
        torch.testing.assert_allclose(actual_ratio,
                                      drop_ratio,
                                      rtol=0.01,
                                      atol=0.01)

    dropout_training_harness(dropout_op, x, check_ratio)


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_dropout2d_training():
    drop_ratio = 0.8
    dropout_op = torch.nn.Dropout2d(drop_ratio)

    # Input size needs to be large enough for convergence to expected dropout ratio
    N = 40
    C = 50
    num_channels = torch.as_tensor(N * C, dtype=torch.float)
    sz = [N, C, 3, 4]
    x = torch.ones(sz, dtype=torch.float)

    def check_ratio(poptorch_out):
        channel_mask = (poptorch_out == 0).all(-1).all(-1)
        actual_ratio = channel_mask.sum() / num_channels
        torch.testing.assert_allclose(actual_ratio,
                                      drop_ratio,
                                      rtol=0.01,
                                      atol=0.01)

    dropout_training_harness(dropout_op, x, check_ratio)


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_dropout3d_training():
    drop_ratio = 0.6
    dropout_op = torch.nn.Dropout3d(drop_ratio)

    # Input size needs to be large enough for convergence to expected dropout ratio
    N = 40
    C = 50
    num_channels = torch.as_tensor(N * C, dtype=torch.float)
    sz = [N, C, 3, 3, 3]
    x = torch.ones(sz, dtype=torch.float)

    def check_ratio(poptorch_out):
        channel_mask = (poptorch_out == 0).all(-1).all(-1).all(-1)
        actual_ratio = channel_mask.sum() / num_channels
        torch.testing.assert_allclose(actual_ratio,
                                      drop_ratio,
                                      rtol=0.01,
                                      atol=0.01)

    dropout_training_harness(dropout_op, x, check_ratio)


def test_embedding():
    model = torch.nn.Embedding(10, 3)
    x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])

    # Run on CPU.
    nativeOut = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    assert nativeOut.size() == poptorch_out.size()
    assert torch.equal(nativeOut, poptorch_out)


def test_pixel_shuffle():
    model = torch.nn.PixelShuffle(3)
    x = torch.randn(2, 18, 4, 4)

    # Run on CPU.
    nativeOut = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    assert nativeOut.size() == poptorch_out.size()
    assert torch.equal(nativeOut, poptorch_out)
