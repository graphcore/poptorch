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


def op_harness(trace_model, op, inputs, inference_test_fn=None):
    if inference_test_fn is None:
        inference_test_fn = lambda native_out, poptorch_out: helpers.assert_allclose(
            expected=native_out, actual=poptorch_out)

    opts = poptorch.Options().randomSeed(42)
    opts.Jit.traceModel(trace_model)
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


@pytest.mark.parametrize("scale_factor", [2, 3.5, 5.00001, 5.12498])
@pytest.mark.parametrize("input_shape", [(1, 2, 8), (2, 2, 2, 8),
                                         (2, 3, 4, 2, 8)])
@pytest.mark.parametrize("trace_model", [True, False])
def test_upsample_nearest(scale_factor, input_shape, trace_model):
    if not trace_model:
        pytest.skip("TODO(T51159): AssertionError: Tensor-likes are not close")
    torch.manual_seed(42)
    op = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
    x = torch.randn(*input_shape)
    op_harness(trace_model, op, [x])


@pytest.mark.parametrize("trace_model", [True, False])
def test_downsample_nearest(trace_model):
    if not trace_model:
        pytest.skip("TODO(T51159): AssertionError: Tensor-likes are not close")
    torch.manual_seed(42)
    # test case from T44610
    op = torch.nn.Upsample(scale_factor=0.435714, mode="nearest")
    x = torch.randn(1, 2, 14, 14)
    op_harness(trace_model, op, [x])


# TODO(T43375): replace scale factor 5 with 3.5
@pytest.mark.parametrize("scale_factor", [2, 5])
@pytest.mark.parametrize("input_shape", [(1, 2, 3, 4), (2, 2, 2, 8)])
@pytest.mark.parametrize("trace_model", [True, False])
def test_upsample_bilinear_factor(scale_factor, input_shape, trace_model):
    torch.manual_seed(42)
    op = torch.nn.Upsample(scale_factor=scale_factor, mode="bilinear")
    x = torch.randn(*input_shape)
    op_harness(trace_model, op, [x])


@pytest.mark.parametrize("shapes", [[(1, 2, 3, 4),
                                     (6, 8)], [(2, 2, 2, 8), (7, 28)]])
@pytest.mark.parametrize("trace_model", [True, False])
def test_upsample_bilinear_factor_shapes(shapes, trace_model):
    torch.manual_seed(42)
    op = torch.nn.Upsample(size=shapes[1], mode="bilinear")
    x = torch.randn(*shapes[0])
    op_harness(trace_model, op, [x])


@pytest.mark.parametrize("shape", [(2, 2, 14, 14)])
@pytest.mark.parametrize("trace_model", [True, False])
def test_upsample_bicubic(shape, trace_model):
    if not trace_model:
        pytest.skip("TODO(T57195): Tensor-likes are not close")

    torch.manual_seed(42)
    model = torch.nn.Upsample(scale_factor=0.4357, mode='bicubic')
    x = torch.randn(*shape)

    # Run on CPU.
    native_out = model(x)

    # Run on IPU.
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    poptorch_out = poptorch_model(x)

    helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.parametrize("mode, input_shape", [("linear", (1, 2, 3)),
                                               ("trilinear", (1, 2, 3, 4, 5))])
@pytest.mark.parametrize("trace_model", [True, False])
def test_unsupported_upsample(mode, input_shape, trace_model):
    if not trace_model:
        pytest.skip("TODO(T57195): refcount_.load() == 0")
    torch.manual_seed(42)
    scale_factor = 2
    model = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)
    x = torch.randn(*input_shape)

    # Run on IPU.
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    with pytest.raises(poptorch.Error, match="only 'nearest' is supported"):
        poptorch_model(x)


@pytest.mark.parametrize("trace_model", [True, False])
def test_linear(trace_model):
    torch.manual_seed(42)
    model = torch.nn.Linear(20, 30)
    x = torch.randn(128, 20)

    # Run on CPU.
    native_out = model(x)

    # Run on IPU.
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    poptorch_out = poptorch_model(x)

    assert native_out.size() == poptorch_out.size()
    helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.parametrize("include_bias", include_bias)
@pytest.mark.parametrize("input_feature_shapes", input_feature_shapes)
@pytest.mark.parametrize("trace_model", [True, False])
def test_bilinear(include_bias, input_feature_shapes, trace_model):
    if not trace_model:
        pytest.skip("TODO(T51159): AssertionError: Tensor-likes are not close")
    torch.manual_seed(42)
    op = torch.nn.Bilinear(20, 30, 40, bias=include_bias)
    shape1 = input_feature_shapes['x1']
    shape2 = input_feature_shapes['x2']
    x1 = torch.randn(8, *shape1, 20)
    x2 = torch.randn(8, *shape2, 30)
    op_harness(trace_model, op, [x1, x2])


@pytest.mark.parametrize("trace_model", [True, False])
def test_identity(trace_model):
    torch.manual_seed(42)
    op = torch.nn.Identity(20, 30, 40)
    x = torch.randn(128, 20)
    op_harness(trace_model, op, [x])


dropout_ops = [torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d]


@pytest.mark.parametrize("dropout_op", dropout_ops)
@pytest.mark.parametrize("trace_model", [True, False])
def test_dropout_inference(dropout_op, trace_model):
    torch.manual_seed(42)
    model = dropout_op()
    model.eval()

    x = torch.randn(128, 20)

    # Run on CPU.
    native_out = model(x)

    # Run on IPU.
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    poptorch_out = poptorch_model(x)

    msg = f"{dropout_op.__name__} in inference session should equal identity."
    helpers.assert_allequal(expected=native_out, actual=poptorch_out, msg=msg)


@pytest.mark.parametrize("dropout_op", dropout_ops)
@pytest.mark.parametrize("trace_model", [True, False])
def test_dropout_eval_during_training(dropout_op, trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): 'popart_exception': Could not find loss tensor '' "
            "in main graph tensors")
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
    opts.Jit.traceModel(trace_model)

    class ModelWithLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = dropout
            self.loss = torch.nn.L1Loss()

        def forward(self, data, target):
            out = self.dropout(data)
            loss = self.loss(out, target)
            return out, loss

    model = ModelWithLoss()
    poptorch_model = poptorch.trainingModel(model, options=opts)
    dummy_label = torch.zeros_like(x)
    poptorch_out, _ = poptorch_model(x, dummy_label)

    assert native_out.size() == poptorch_out.size()
    msg = f"{dropout_op.__name__} should equal identity."
    helpers.assert_allequal(expected=x, actual=poptorch_out, msg=msg)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_dropout_training(trace_model):
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

    op_harness(trace_model, dropout_op, [x], check_ratio)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_dropout2d_training(trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::bernoulli")
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

    op_harness(trace_model, dropout_op, [x], check_ratio)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_dropout3d_training(trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::bernoulli")
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

    op_harness(trace_model, dropout_op, [x], check_ratio)


@pytest.mark.parametrize("trace_model", [True, False])
def test_embedding(trace_model):
    model = torch.nn.Embedding(10, 3)
    x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])

    # Run on CPU.
    native_out = model(x)

    # Run on IPU.
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    poptorch_out = poptorch_model(x)

    assert native_out.size() == poptorch_out.size()
    helpers.assert_allequal(expected=native_out, actual=poptorch_out)


# pylint: disable=unsubscriptable-object
@pytest.mark.parametrize("trace_model", [True, False])
def test_embedding_padding_idx(trace_model):
    if not trace_model:
        pytest.skip("TODO(T51159): ValueError: not enough values to unpack "
                    "(expected 2, got 1)")

    class TestEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.embedding = torch.nn.Embedding(10, 4, padding_idx=0)

        def forward(self, x):
            y = self.embedding(x)
            loss = poptorch.identity_loss(y.sum(), "none")
            return y, loss

    model = TestEmbedding()
    # pylint:disable=unsubscriptable-object
    x = torch.arange(0, model.embedding.weight.shape[0])
    y, loss = model(x)
    loss.backward()
    grad = model.embedding.weight.grad

    options = poptorch.Options()
    options.anchorTensor("grad_embedding", "Gradient___model.embedding.weight")
    options.Jit.traceModel(trace_model)
    pop_model = poptorch.trainingModel(TestEmbedding(), options=options)
    pop_y, pop_loss = pop_model(x)
    pop_grad = pop_model.getAnchoredTensor("grad_embedding")

    helpers.assert_allclose(actual=pop_y, expected=y)
    helpers.assert_allclose(actual=pop_loss, expected=loss)
    helpers.assert_allclose(actual=pop_grad, expected=grad)


@pytest.mark.parametrize("mode", ["max", "mean", "sum"])
@pytest.mark.parametrize("trace_model", [True, False])
def test_embedding_bag(mode, trace_model):
    torch.manual_seed(0)
    model = torch.nn.EmbeddingBag(10, 3, mode=mode)
    x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    cpu_out = model(x)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    pop_model = poptorch.inferenceModel(model, options)
    pop_out = pop_model(x)
    helpers.assert_allclose(actual=pop_out, expected=cpu_out)


@pytest.mark.parametrize("trace_model", [True, False])
def test_embedding_bag_per_sample_weights(trace_model):
    if not trace_model:
        pytest.skip("TODO(T57195): Could not find tensor")

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
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    pop_model = poptorch.inferenceModel(model, options)
    pop_out = pop_model(x)
    helpers.assert_allclose(actual=pop_out, expected=cpu_out)


@pytest.mark.parametrize("mode", ["max", "mean", "sum"])
@pytest.mark.parametrize("trace_model", [True, False])
def test_embedding_bag_include_last_offset(mode, trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T57195): Unsupported aten::embedding_bag operation: " +
            "offsets tensor must be a constant")

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
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    pop_model = poptorch.inferenceModel(model, options)
    pop_out = pop_model(x)
    helpers.assert_allclose(actual=pop_out, expected=cpu_out)


@pytest.mark.parametrize("trace_model", [True, False])
def test_pixel_shuffle(trace_model):
    torch.manual_seed(42)
    op = torch.nn.PixelShuffle(3)
    x = torch.randn(2, 18, 4, 4)
    op_harness(trace_model, op, [x])


@pytest.mark.parametrize("params", [(2, 2, 1, 1, 1, 1), (3, 2, 1, 1, 1, 1),
                                    (2, 4, 1, 1, 1, 1), (2, 2, 2, 1, 1, 1),
                                    (2, 2, 1, 3, 1, 1), (2, 2, 1, 1, 3, 1),
                                    (2, 2, 1, 1, 1, 4)])
@pytest.mark.parametrize("trace_model", [True, False])
# Tests aten::im2col
def test_unfold(params, trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::im2col")
    (kernel_size_x, kernel_size_y, dilation_x, dilation_y, stride_x,
     stride_y) = params
    padding = 2
    y_in = 19
    x_in = 23

    unfold_layer = torch.nn.Unfold(kernel_size=(kernel_size_y, kernel_size_x),
                                   dilation=(dilation_y, dilation_x),
                                   padding=padding,
                                   stride=(stride_y, stride_x))

    numel_y = (y_in + 2 * padding - dilation_y *
               (kernel_size_y - 1) - 1) // stride_y + 1
    numel_x = (x_in + 2 * padding - dilation_x *
               (kernel_size_x - 1) - 1) // stride_x + 1
    numel = numel_y * numel_x

    linear_layer = torch.nn.Linear(numel, numel)
    combined = torch.nn.Sequential(unfold_layer, linear_layer)

    inputs = [torch.rand(1, 1, y_in, x_in)]

    op_harness(trace_model, combined, inputs)


@pytest.mark.parametrize("params", [(2, 2, 1, 1, 1, 1), (3, 2, 1, 1, 1, 1),
                                    (2, 4, 1, 1, 1, 1), (2, 2, 2, 1, 1, 1),
                                    (2, 2, 1, 3, 1, 1), (2, 2, 1, 1, 3, 1),
                                    (2, 2, 1, 1, 1, 3)])
# Tests aten::col2im
@pytest.mark.parametrize("trace_model", [True, False])
def test_fold(params, trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::col2im")
    (kernel_size_x, kernel_size_y, dilation_x, dilation_y, stride_x,
     stride_y) = params

    torch.manual_seed(42)
    orig_input = torch.rand(2, 3, 11, 13)

    # unfold the input to provide an input to fold
    unfold_args = {
        "kernel_size": (kernel_size_y, kernel_size_x),
        "dilation": (dilation_y, dilation_x),
        "padding": (0, 0),
        "stride": (stride_y, stride_x)
    }
    unfold = torch.nn.Unfold(**unfold_args)
    unfolded = unfold(orig_input)

    unfold_args["output_size"] = orig_input.shape[2:]

    op = torch.nn.Fold(**unfold_args)
    op_harness(trace_model, op, [unfolded])


# Tests aten::col2im with padding
@pytest.mark.parametrize("stride_x", [1, 3])
@pytest.mark.parametrize("stride_y", [1, 3])
@pytest.mark.parametrize("trace_model", [True, False])
def test_fold_with_padding(stride_x, stride_y, trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::col2im")
    torch.manual_seed(42)

    orig_input = torch.rand(2, 2, 11, 13)

    # unfold the input to provide an input to fold
    unfold_args = {
        "kernel_size": (2, 2),
        "dilation": (1, 1),
        "padding": (2, 2),
        "stride": (stride_y, stride_x)
    }
    unfold = torch.nn.Unfold(**unfold_args)
    unfolded = unfold(orig_input)

    # Since it is zero-padded, add a little to every value
    unfolded += 1.0

    unfold_args["output_size"] = orig_input.shape[2:]

    op = torch.nn.Fold(**unfold_args)
    op_harness(trace_model, op, [unfolded])


@pytest.mark.parametrize("dim", [0, 1, None])
@pytest.mark.parametrize("trace_model", [True, False])
def test_weight_norm(trace_model, dim):
    if not trace_model:
        pytest.skip(
            "TODO(T57195): Only Tensors created explicitly by the user " +
            "(graph leaves) support the deepcopy protocol at the moment")

    torch.manual_seed(42)

    x = torch.randn(10)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            lin = torch.nn.Linear(10, 5)
            # Wrap the linear layer with a weight_norm - This should
            # decompose "weight" into "weight_v" and "weight_g"
            self.lin = torch.nn.utils.weight_norm(lin, "weight", dim)

        def forward(self, x):
            x = self.lin(x)
            return x, poptorch.identity_loss(x**2, reduction="sum")

    model = Model()
    weight_v_before = model.lin.weight_v.detach().clone()
    weight_g_before = model.lin.weight_g.detach().clone()

    native_out, _ = model(x)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options)

    poptorch_out, _ = poptorch_model(x)

    helpers.assert_allclose(expected=native_out, actual=poptorch_out)

    tensor_names = poptorch_model.getTensorNames()
    decomposed_tensors = ["weight_v", "weight_g"]

    # Check that both decomposed tensors exist in the graph
    assert all(f"model.lin.{t}" in tensor_names for t in decomposed_tensors)
    # Check that they also exist in the backward graph
    assert all(f"UpdatedVar___model.lin.{t}" in tensor_names
               for t in decomposed_tensors)

    # Ensure that the original weight tensor does NOT exist -
    # autograd should be performed with respect to the decomposed tensors
    # only
    assert "model.lin.weight" not in tensor_names
    assert "UpdatedVar___model.lin.weight" not in tensor_names

    n = 3
    # Run a few more times to ensure that the decomposed weights are being
    # updated each time
    for i in range(n):
        weight_v_after = poptorch_model.lin.weight_v.detach().clone()
        weight_g_after = poptorch_model.lin.weight_g.detach().clone()

        # Ensure the decomposed weights changed since the previous iteration
        assert not torch.allclose(weight_v_before, weight_v_after)
        assert not torch.allclose(weight_g_before, weight_g_after)

        # Prepare for the next iteration
        if i != n - 1:
            weight_v_before = weight_v_after
            weight_g_before = weight_g_after

            poptorch_model(x)
