// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

TORCH_LIBRARY_IMPL(aten, PrivateUse2, m) {
  m.impl("copy_", PTC(poptorch::copyInplace));
  m.impl("_to_copy", PTC(poptorch::toCopy));
  m.impl("empty.memory_format", PTC(poptorch::emptyMemoryFormat));
  m.impl("empty.out", PTC(poptorch::emptyOut));
  m.impl("empty_strided", PTC(poptorch::emptyStrided));
  m.impl("detach", PTC(poptorch::detach));
  m.impl("convolution", PTC_BOXED(poptorch::fallback));
  m.impl("transpose.int", PTC_BOXED(poptorch::fallback));
  m.impl("layer_norm", PTC_BOXED(poptorch::fallback));
  m.impl("expand", PTC_BOXED(poptorch::fallback));
  m.impl("dropout", PTC_BOXED(poptorch::fallback));
  m.impl("avg_pool2d.out", PTC_BOXED(poptorch::fallback));
  m.impl("avg_pool3d.out", PTC_BOXED(poptorch::fallback));
  m.impl("max_pool1d", PTC_BOXED(poptorch::fallback));
  m.impl("max_pool2d", PTC_BOXED(poptorch::fallback));
  m.impl("max_pool3d", PTC_BOXED(poptorch::fallback));
  m.impl("adaptive_avg_pool1d", PTC_BOXED(poptorch::fallback));
  m.impl("adaptive_avg_pool2d", PTC_BOXED(poptorch::fallback));
  m.impl("adaptive_avg_pool3d", PTC_BOXED(poptorch::fallback));
  m.impl("var", PTC_BOXED(poptorch::fallback));
  m.impl("var_mean", PTC_BOXED(poptorch::fallback));
  m.impl("std", PTC_BOXED(poptorch::fallback));
  m.impl("std_mean", PTC_BOXED(poptorch::fallback));
  m.impl("trunc", PTC_BOXED(poptorch::fallback));
  m.impl("min", PTC_BOXED(poptorch::fallback));
  m.impl("minimum", PTC_BOXED(poptorch::fallback));
  m.impl("max", PTC_BOXED(poptorch::fallback));
  m.impl("maximum", PTC_BOXED(poptorch::fallback));
  m.impl("argsort", PTC_BOXED(poptorch::fallback));
  m.impl("one_hot", PTC_BOXED(poptorch::fallback));
  m.impl("all", PTC_BOXED(poptorch::fallback));
  m.impl("any", PTC_BOXED(poptorch::fallback));

  // If we don't intercept this op, it will be decomposed to as_strided
  // which is harder to handle.
  m.impl("slice.Tensor", PTC_BOXED(poptorch::fallback));

  // If we don't intercept this op, it will be decomposed to as_strided
  // which is harder to handle.
  m.impl("squeeze.dim", PTC_BOXED(poptorch::fallback));
  m.impl("squeeze_.dim", PTC_BOXED(poptorch::fallback));

  // If we don't intercept this op, it will be decomposed to as_strided
  // which is harder to handle.
  m.impl("unsqueeze", PTC_BOXED(poptorch::fallback));

  // If we don't intercept this op, it will be decomposed to as_strided
  // which is harder to handle.
  m.impl("permute", PTC_BOXED(poptorch::fallback));

  // If we don't intercept this op, it will be decomposed to as_strided
  // which is harder to handle.
  m.impl("select.int", PTC_BOXED(poptorch::fallback));

  // Ideally, we would use the native cpu function but have an equivalent
  // to the "if (self.is_mkldnn()) {" for IPU tensors. But we can instead
  // overwrite and run reshape here.
  m.impl("reshape", PTC_BOXED(poptorch::fallback));

  m.impl("constant_pad_nd", PTC_BOXED(poptorch::fallback));
  m.impl("binary_cross_entropy_with_logits", PTC_BOXED(poptorch::fallback));
  m.impl("binary_cross_entropy_with_logits_backward", PTC_BOXED(poptorch::fallback));
}
