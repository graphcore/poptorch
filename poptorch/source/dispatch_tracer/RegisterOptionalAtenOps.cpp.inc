// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

TORCH_LIBRARY_IMPL(aten, PrivateUse2, m) {
  m.impl("copy_", &poptorch::copyInplace);

#if TORCH_MINOR_VERSION >= 10
  m.impl("_to_copy", &poptorch::toCopy);
#endif

  m.impl("empty.memory_format", &poptorch::emptyMemoryFormat);
  m.impl("empty.out", &poptorch::emptyOut);
  m.impl("empty_strided", &poptorch::emptyStrided);

  m.impl("detach", &poptorch::detach);

  m.impl("transpose.int",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("layer_norm",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("expand",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("dropout",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("avg_pool2d.out",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("avg_pool3d.out",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("max_pool1d",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("max_pool2d",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("max_pool3d",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("adaptive_avg_pool1d",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("adaptive_avg_pool2d",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("adaptive_avg_pool3d",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("var",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("var_mean",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("std",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("std_mean",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  // If we don't intercept this op, it will be decomposed to as_strided
  // which is harder to handle.
  m.impl("slice.Tensor",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  // If we don't intercept this op, it will be decomposed to as_strided
  // which is harder to handle.
  m.impl("squeeze.dim",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("squeeze_.dim",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  // If we don't intercept this op, it will be decomposed to as_strided
  // which is harder to handle.
  m.impl("unsqueeze",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  // If we don't intercept this op, it will be decomposed to as_strided
  // which is harder to handle.
  m.impl("permute",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  // If we don't intercept this op, it will be decomposed to as_strided
  // which is harder to handle.
  m.impl("select.int",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  // Ideally, we would use the native cpu function but have an equivalent
  // to the "if (self.is_mkldnn()) {" for IPU tensors. But we can instead
  // overwrite and run reshape here.
  m.impl("reshape",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("constant_pad_nd",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("binary_cross_entropy_with_logits",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());

  m.impl("binary_cross_entropy_with_logits_backward",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
}
