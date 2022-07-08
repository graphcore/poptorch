// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

//TODO(T59880) rename XLA -> IPU
TORCH_LIBRARY_IMPL(aten, XLA, m) {
  m.impl("copy_", PTC(poptorch::copyInplace));
  m.impl("_local_scalar_dense", PTC(poptorch::localScalarDense));

  m.impl("empty.memory_format", PTC(poptorch::emptyMemoryFormat));
  m.impl("empty_strided", PTC(poptorch::emptyStrided));

  m.impl("convolution_overrideable", PTC_BOXED(poptorch::fallback));
  m.impl("convolution_backward_overrideable", PTC_BOXED(poptorch::fallback));
  m.impl("transpose.int", PTC_BOXED(poptorch::fallback));
  m.impl("layer_norm", PTC_BOXED(poptorch::fallback));
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

  // If we don't intercept this op, it will be decomposed to as_strided
  // which is harder to handle.
  m.impl("transpose_", PTC_BOXED(poptorch::fallback));

  // If we don't intercept this op, it will be call the native implementation
  // of alias, which will try to access our tensor storage.
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
  m.impl("triplet_margin_loss", PTC_BOXED(poptorch::fallback));
}
