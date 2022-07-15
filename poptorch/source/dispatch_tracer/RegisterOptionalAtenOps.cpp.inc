// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

//TODO(T59880) rename XLA -> IPU
TORCH_LIBRARY_IMPL(aten, XLA, m) {
  m.impl("copy_", PTC(poptorch::copyInplace));
  m.impl("_local_scalar_dense", PTC(poptorch::localScalarDense));
  m.impl("item", PTC(poptorch::item));

  m.impl("empty.memory_format", PTC(poptorch::emptyMemoryFormat));
  m.impl("empty_strided", PTC(poptorch::emptyStrided));

  m.impl("convolution_overrideable", PTC_BOXED(poptorch::fallback));
  m.impl("convolution_backward_overrideable", PTC_BOXED(poptorch::fallback));
  m.impl("transpose.int", PTC_BOXED(poptorch::fallback));
  m.impl("expand", PTC_BOXED(poptorch::fallback));
  m.impl("gather", PTC_BOXED(poptorch::fallback));
  m.impl("dropout", PTC_BOXED(poptorch::fallback));
  m.impl("avg_pool2d.out", PTC_BOXED(poptorch::fallback));
  m.impl("avg_pool3d.out", PTC_BOXED(poptorch::fallback));
  m.impl("max_pool1d", PTC_BOXED(poptorch::fallback));
  m.impl("max_pool2d", PTC_BOXED(poptorch::fallback));
  m.impl("max_pool3d", PTC_BOXED(poptorch::fallback));
  m.impl("adaptive_avg_pool1d", PTC_BOXED(poptorch::fallback));
  m.impl("adaptive_avg_pool2d", PTC_BOXED(poptorch::fallback));
  m.impl("adaptive_avg_pool3d", PTC_BOXED(poptorch::fallback));
  m.impl("trunc", PTC_BOXED(poptorch::fallback));
  m.impl("min", PTC_BOXED(poptorch::fallback));
  m.impl("minimum", PTC_BOXED(poptorch::fallback));
  m.impl("max", PTC_BOXED(poptorch::fallback));
  m.impl("maximum", PTC_BOXED(poptorch::fallback));
  m.impl("argsort", PTC_BOXED(poptorch::fallback));
  m.impl("one_hot", PTC_BOXED(poptorch::fallback));
  m.impl("all", PTC_BOXED(poptorch::fallback));
  m.impl("any", PTC_BOXED(poptorch::fallback));

  // Intercept alias to circumvent issues accessing tensor storage directly
  m.impl("alias", PTC_BOXED(poptorch::fallback));

  // Needed due to "CompositeImplicitAutograd"
  m.impl("native_group_norm",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("native_layer_norm",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

// If we don't intercept these ops, they will be decomposed into
// as_strided which is harder to handle.
  m.impl("slice.Tensor", PTC_BOXED(poptorch::fallback));
  m.impl("squeeze", PTC_BOXED(poptorch::fallback));
  m.impl("squeeze_", PTC_BOXED(poptorch::fallback));
  m.impl("squeeze.dim", PTC_BOXED(poptorch::fallback));
  m.impl("squeeze_.dim", PTC_BOXED(poptorch::fallback));
  m.impl("unsqueeze", PTC_BOXED(poptorch::fallback));
  m.impl("permute", PTC_BOXED(poptorch::fallback));
  m.impl("select.int", PTC_BOXED(poptorch::fallback));
  m.impl("transpose_", PTC_BOXED(poptorch::fallback));

  // If we don't intercept this op, it will be decomposed into
  // _index_put_impl_, which exposes unnecessary implementation
  // details
  m.impl("index_put_", PTC_BOXED(poptorch::fallback));

  // If we don't catch these, PyTorch will try to call aten::resize_ on the
  // result which is not supported.
  m.impl("frobenius_norm.out", PTC_BOXED(poptorch::fallback));
  m.impl("frobenius_norm.dim", PTC_BOXED(poptorch::fallback));

  // Use our own repeat op
  m.impl("repeat", PTC_BOXED(poptorch::fallback));

  m.impl("constant_pad_nd", PTC_BOXED(poptorch::fallback));
  m.impl("binary_cross_entropy_with_logits", PTC_BOXED(poptorch::fallback));
  m.impl("binary_cross_entropy_with_logits_backward", PTC_BOXED(poptorch::fallback));

  // If we don't catch it here, PyTorch will decompose bilinear into an enormous
  // number of ops, which will result in an all-zeros output.
  m.impl("bilinear", PTC_BOXED(poptorch::fallback));

  // Loss functions: these are needed for popart, so that we can mark the loss
  // tensor (see `IsLoss`); otherwise, the op will get decomposed by PyTorch.
  m.impl("cosine_embedding_loss", PTC_BOXED(poptorch::fallback));
  m.impl("ctc_loss.IntList", PTC_BOXED(poptorch::fallback));
  m.impl("ctc_loss.Tensor", PTC_BOXED(poptorch::fallback));
  m.impl("hinge_embedding_loss", PTC_BOXED(poptorch::fallback));
  m.impl("kl_div", PTC_BOXED(poptorch::fallback));
  m.impl("l1_loss.out", PTC_BOXED(poptorch::fallback));
  m.impl("margin_ranking_loss", PTC_BOXED(poptorch::fallback));
  m.impl("poisson_nll_loss", PTC_BOXED(poptorch::fallback));
  m.impl("soft_margin_loss.out", PTC_BOXED(poptorch::fallback));
  m.impl("triplet_margin_loss", PTC_BOXED(poptorch::fallback));

  // Scatter: By default, PyTorch's handler will fail if the index tensor isn't
  // a tensor of int64s (see `scatter_gather_dtype_check` in PyTorch) -- ours
  // will have been coerced to int32s.
  m.impl("scatter.src", PTC_BOXED(poptorch::fallback));
  m.impl("scatter.src_out", PTC_BOXED(poptorch::fallback));
  m.impl("scatter_.src", PTC_BOXED(poptorch::fallback));

  m.impl("scatter.value", PTC_BOXED(poptorch::fallback));
  m.impl("scatter.value_out", PTC_BOXED(poptorch::fallback));
  m.impl("scatter_.value", PTC_BOXED(poptorch::fallback));

  m.impl("scatter.reduce", PTC_BOXED(poptorch::fallback));
  m.impl("scatter.reduce_out", PTC_BOXED(poptorch::fallback));
  m.impl("scatter_.reduce", PTC_BOXED(poptorch::fallback));

  m.impl("scatter.value_reduce", PTC_BOXED(poptorch::fallback));
  m.impl("scatter.value_reduce_out", PTC_BOXED(poptorch::fallback));
  m.impl("scatter_.value_reduce", PTC_BOXED(poptorch::fallback));

  m.impl("scatter_add", PTC_BOXED(poptorch::fallback));
  m.impl("scatter_add.out", PTC_BOXED(poptorch::fallback));
  m.impl("scatter_add_", PTC_BOXED(poptorch::fallback));
}
