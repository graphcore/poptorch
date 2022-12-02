// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <ATen/MetaFunctions.h>

namespace poptorch::meta {

std::tuple<at::Tensor,at::Tensor> nllLoss2dForward(const at::Tensor &self,
                                                      const at::Tensor &/*target*/,
                                                      const c10::optional<at::Tensor> &/*weight*/,
                                                      int64_t reduction,
                                                      int64_t /*ignore_index*/) {
  // If reduction is none, the shape is the the input without number of
  // classes, which is the second element, i.e. (N, C, ...) to (N, ...)
  // except in the case of a 1D input (C) when it is ().
  std::vector<std::int64_t> shape;
  if (reduction == 0){
    shape = std::vector<std::int64_t>(self.sizes().begin(), self.sizes().end());
    if(shape.size() == 1) {
      shape.clear();
    } else {
      ERROR_ON(shape.size() < 2);
      shape.erase(shape.begin() + 1);
    }
  }
  at::Tensor output = at::meta::empty(shape, self.scalar_type());
  at::Tensor total_weight = at::meta::empty({}, self.scalar_type());
  return {output, total_weight};
}

at::Tensor ctcLoss(const at::Tensor &log_probs, const at::Tensor &/*targets*/,
                      at::IntArrayRef /*input_lengths*/, at::IntArrayRef /*target_lengths*/,
                      int64_t /*blank*/, int64_t reduction, bool /*zero_infinity*/) {
  std::vector<std::int64_t> shape;
  if (reduction == 0 && log_probs.sizes().size() == 3) {
    shape = {log_probs.sizes()[1]};
  }
  return at::meta::empty(shape, log_probs.scalar_type());
}

at::Tensor ctcLossTensor(const at::Tensor &log_probs, const at::Tensor &/*targets*/,
                            const at::Tensor &/*input_lengths*/, const at::Tensor &/*target_lengths*/,
                            int64_t /*blank*/, int64_t reduction, bool /*zero_infinity*/) {
  // If no reduction, get the batch size; from docs, this will be
  // `log_probs`' second dimension if it's 3D.
  std::vector<std::int64_t> shape;
  if (reduction == 0 && log_probs.sizes().size() == 3) {
    shape = {log_probs.sizes()[1]};
  }
  return at::meta::empty(shape, log_probs.scalar_type());
}

at::Tensor median(const at::Tensor &self) {
  return at::meta::empty({}, self.scalar_type());
}

std::tuple<at::Tensor,at::Tensor> medianDim(const at::Tensor &self,
                                            int64_t dim, bool keepdim) {
  std::vector<std::int64_t> shape = self.sizes().vec();
  dim = dim < 0 ? dim + self.sizes().size() : dim;

  if (!shape.empty()) {
    if (keepdim) {
      shape[dim] = 1;
    } else {
      shape.erase(shape.begin() + dim);
    }
  }

  auto values = at::meta::empty(shape, self.scalar_type());
  auto indices = at::meta::empty(shape, c10::ScalarType::Long);
  return {values, indices};
}

at::Tensor countNonzero(const at::Tensor &self, at::IntArrayRef dim) {
  auto dim_vec = dim.vec();
  for (auto &d : dim_vec) {
    d = d < 0 ? d + self.sizes().size() : d;
  }

  std::vector<std::int64_t> shape = {1};
  if (dim.size() > 0) {
    shape = self.sizes().vec();
    auto sorted_dims = dim_vec;
    std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<>{});

    ERROR_ON_MSG(std::adjacent_find(sorted_dims.begin(), sorted_dims.end()) != sorted_dims.end(), 
                  "The dimensions to count must be unique");

    for (auto d : sorted_dims) {
      shape.erase(shape.begin() + d);
    }
  }

  return at::meta::empty(shape, self.scalar_type());
}

at::Tensor oneHot(const at::Tensor &self, int64_t num_classes) {
  ERROR_ON_MSG(num_classes == -1, "OneHot num classes must be specified and must be constant.");

  auto shape = self.sizes().vec();
  shape.push_back(num_classes);
  return at::meta::empty(shape, self.scalar_type());
}

at::Tensor upsampleNearest3d(const at::Tensor &input, at::OptionalSymIntArrayRef output_size,
                              c10::optional<at::ArrayRef<double>> scale_factors) {
  ERROR_ON_MSG(!scale_factors && !output_size,
               "Must specify either output_size or scale_factors, but not both.");
  const auto input_shape = input.sizes().vec();
  std::vector<int64_t> actual_output_size;
  if (output_size.has_value()) {
    ERROR_ON_MSG(scale_factors,
                 "Must specify either output_size or scale_factors, but not both.");
    actual_output_size.reserve(output_size->size());
    for (auto i : output_size.value()) {
      actual_output_size.push_back(i.as_int_unchecked());
    }
  }
  else if (scale_factors.has_value()) {
    std::transform(scale_factors->begin(), scale_factors->end(),
                    input_shape.end() - scale_factors->size(),
                    std::back_inserter(actual_output_size),
                    [](double sf, std::int64_t shape) {
                        return static_cast<int64_t>(static_cast<double>(shape) * sf);
                    });
  }

  ERROR_ON_MSG(actual_output_size.size() > input_shape.size(),
              "The number of dimensions of the input (" + std::to_string(input_shape.size()) +
              ") must be more than the number of dimensions in the output (" +
              std::to_string(actual_output_size.size()) + ")");

  std::vector<std::int64_t> shape(input_shape.begin(), input_shape.end() - actual_output_size.size());
  shape.insert(shape.end(), actual_output_size.begin(), actual_output_size.end());
  return at::meta::empty(shape, input.scalar_type());
}

at::Tensor maxPool3d(const at::Tensor &self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  std::vector<std::int64_t> input_shape = self.sizes().vec();

  ERROR_ON_MSG(input_shape.size() != kernel_size.size() + 1 &&
                input_shape.size() != kernel_size.size() + 2,
                "The kernel size (" << kernel_size.size() <<
                ") must be 1 or 2 less than the input rank ("
                << input_shape.size() << ")");
  ERROR_ON(kernel_size.size() != stride.size());
  ERROR_ON(kernel_size.size() != padding.size());
  ERROR_ON(kernel_size.size() != dilation.size());

  const size_t offset = (input_shape.size() == kernel_size.size() + 1) ? 1 : 2;

  for (auto s = 0u; s < kernel_size.size(); s++) {
    double tmp = (input_shape[offset + s] + 2. * padding[s] - dilation[s] * (kernel_size[s] - 1.) - 1.) / stride[s] + 1.;
    if (ceil_mode) {
      input_shape[offset + s] = std::ceil(tmp);
    } else {
      input_shape[offset + s] = std::floor(tmp);
    }
  }
  return at::meta::empty(input_shape, self.scalar_type());
}

at::Tensor nonzero(const at::Tensor &) {
  ERROR("Operations using aten::nonzero are unsupported because "
        "the output shape is determined by the tensor values. "
        "The IPU cannot support dynamic output shapes.");
}

// torch_scatter
std::tuple<at::Tensor, at::Tensor> scatterMinMax(at::Tensor src,
                                                 at::Tensor /*index*/,
                                                 int64_t dim,
                                                 c10::optional<at::Tensor> out,
                                                 c10::optional<int64_t> dim_size) {
  std::vector<std::int64_t> out_shape = src.sizes().vec();

  dim = dim < 0 ? dim + out_shape.size() : dim;

  if (out) {
    out_shape = out->sizes().vec();
  } else if (dim_size) {
    out_shape[dim] = *dim_size;
  } else {
    ERROR("You must provide either an output parameter or specify dim_size so the output shape may be inferred");
  }

  if (dim_size.has_value()) {
    ERROR_ON_MSG(*dim_size != out_shape[dim], "dim_size expected to be the same as out.shape()[dim]");
  }

  auto output = at::meta::empty(out_shape, src.scalar_type());
  auto argminmax = at::meta::empty(out_shape, c10::ScalarType::Long);
  return {output, argminmax};
}

// poptorch

// dynamic_slice(Tensor self, int dim, Tensor start, int size, int step) -> Tensor
at::Tensor dynamicSlice(const at::Tensor &self, int64_t dim, const at::Tensor &/*start*/,
                        int64_t size, int64_t step) {
  auto shape = self.sizes().vec();
  shape[dim] = (size + (step - 1)) / step;
  return at::meta::empty(shape, self.scalar_type());
}

// custom_operation(Tensor[] inputs, str name, str domain, int domain_version, int num_outputs, Tensor(a!)[] outputs, str attributes) -> Tensor(a!)[]
std::vector<at::Tensor> customOperation(const std::vector<at::Tensor> &/*inputs*/,
                                        const std::string &/*name*/,
                                        const std::string &/*domain*/,
                                        int64_t /*domain_version*/,
                                        int64_t /*num_outputs*/,
                                        const std::vector<at::Tensor> &outputs,
                                        const std::string &/*attributes*/) {
  std::vector<at::Tensor> ret;
  for (const auto &t : outputs) {
    ret.push_back(at::meta::empty(t.sizes(), t.scalar_type()));
  }
  return ret;
}

// ctc_beam_search_decoder(Tensor probs, Tensor lengths, int blank, int beam_width, int top_paths) -> (Tensor, Tensor, Tensor)
std::tuple<at::Tensor, at::Tensor, at::Tensor>
ctcBeamSearchDecoder(const at::Tensor &probs, const at::Tensor &/*lengths*/,
                      int64_t /*blank*/, int64_t /*beam_width*/, int64_t top_paths) {
  ERROR_ON_MSG(probs.sizes().size() != 3,
              "Input probablities tensor must be rank-3 for "
              "`ctc_beam_search_decoder`.");
  const auto input_size = probs.sizes()[0];
  const auto batch_size = probs.sizes()[1];
  auto out_probs = at::meta::empty({batch_size, top_paths}, probs.scalar_type());
  auto out_paths = at::meta::empty({batch_size, top_paths, input_size}, probs.scalar_type());
  return {out_probs, out_probs, out_paths};
}

// identity_loss(Tensor x, str reduction) -> Tensor
at::Tensor identityLoss(const at::Tensor &x, int64_t reduction) {
  constexpr int64_t sum = 0;
  constexpr int64_t mean = 1;
  constexpr int64_t none = 2;
  std::vector<int64_t> sizes;
  switch (reduction) {
  case sum:
  case mean:
    break;
  case none:
    sizes = x.sizes().vec();
    break;
  default:
    ERROR("reduction must be sum (0), mean (1) or none (2)");
  }
  return at::meta::empty(sizes, x.scalar_type());
}

void opWithoutOutputs(const c10::OperatorHandle &/*op*/, c10::Stack *stack) {
  stack->clear();
}

void opReturningFirstArgument(const c10::OperatorHandle &/*op*/, c10::Stack *stack) {
  stack->erase(stack->begin() + 1, stack->end());
}
} // namespace poptorch::meta

TORCH_LIBRARY_IMPL(aten, Meta, m) {
  m.impl("rrelu_with_noise", PTC_BOXED(poptorch::meta::opReturningFirstArgument));

  m.impl("count_nonzero.dim_IntList", PTC(poptorch::meta::countNonzero));
  m.impl("ctc_loss.Tensor", PTC(poptorch::meta::ctcLossTensor));
  m.impl("ctc_loss.IntList", PTC(poptorch::meta::ctcLoss));
  m.impl("max_pool3d", PTC(poptorch::meta::maxPool3d));
  m.impl("median", PTC(poptorch::meta::median));
  m.impl("median.dim", PTC(poptorch::meta::medianDim));
  m.impl("nll_loss2d_forward", PTC(poptorch::meta::nllLoss2dForward));
  m.impl("nonzero", PTC(poptorch::meta::nonzero));
  m.impl("one_hot", PTC(poptorch::meta::oneHot));
  m.impl("upsample_nearest3d.vec", PTC(poptorch::meta::upsampleNearest3d));
}

TORCH_LIBRARY_IMPL(torch_scatter, Meta, m) {
  m.impl("scatter_max", PTC(poptorch::meta::scatterMinMax));
  m.impl("scatter_min", PTC(poptorch::meta::scatterMinMax));
}

TORCH_LIBRARY_IMPL(poptorch, Meta, m) {
  m.impl("push_name_scope", PTC_BOXED(poptorch::meta::opWithoutOutputs));
  m.impl("pop_name_scope", PTC_BOXED(poptorch::meta::opWithoutOutputs));
  m.impl("begin_ipu_block", PTC_BOXED(poptorch::meta::opWithoutOutputs));
  m.impl("end_ipu_block", PTC_BOXED(poptorch::meta::opWithoutOutputs));
  m.impl("start_for_loop", PTC_BOXED(poptorch::meta::opWithoutOutputs));
  m.impl("optimizer_group", PTC_BOXED(poptorch::meta::opWithoutOutputs));
  m.impl("call_cpu_op", PTC_BOXED(poptorch::meta::opWithoutOutputs));
  m.impl("set_attribute", PTC_BOXED(poptorch::meta::opWithoutOutputs));
  m.impl("clear_attribute", PTC_BOXED(poptorch::meta::opWithoutOutputs));
  m.impl("begin_multi_conv", PTC_BOXED(poptorch::meta::opWithoutOutputs));
  m.impl("end_multi_conv", PTC_BOXED(poptorch::meta::opWithoutOutputs));
  
  m.impl("end_cpu_op", PTC_BOXED(poptorch::meta::opReturningFirstArgument));
  m.impl("end_for_loop", PTC_BOXED(poptorch::meta::opReturningFirstArgument));
  m.impl("internal_cast", PTC_BOXED(poptorch::meta::opReturningFirstArgument));
  m.impl("ipu_print_tensor", PTC_BOXED(poptorch::meta::opReturningFirstArgument));
  m.impl("nop", PTC_BOXED(poptorch::meta::opReturningFirstArgument));
  m.impl("recomputation_checkpoint", PTC_BOXED(poptorch::meta::opReturningFirstArgument));
  m.impl("set_available_memory", PTC_BOXED(poptorch::meta::opReturningFirstArgument));
  m.impl("set_matmul_serialization", PTC_BOXED(poptorch::meta::opReturningFirstArgument));
  m.impl("set_overlap_for_input", PTC_BOXED(poptorch::meta::opReturningFirstArgument));
  m.impl("set_overlap_for_output", PTC_BOXED(poptorch::meta::opReturningFirstArgument));

  m.impl("ctc_beam_search_decoder", PTC(poptorch::meta::ctcBeamSearchDecoder));
  m.impl("custom_operation", PTC(poptorch::meta::customOperation));
  m.impl("dynamic_slice", PTC(poptorch::meta::dynamicSlice));
  m.impl("identity_loss", PTC(poptorch::meta::identityLoss));
}

TORCH_LIBRARY_IMPL(poptorch, AutogradMeta, m) {
  m.impl("begin_ipu_block", torch::autograd::autogradNotImplementedFallback());
  m.impl("end_ipu_block", torch::autograd::autogradNotImplementedFallback());
  m.impl("ipu_print_tensor", torch::autograd::autogradNotImplementedFallback());
  m.impl("internal_cast", torch::autograd::autogradNotImplementedFallback());
  m.impl("nop", torch::autograd::autogradNotImplementedFallback());
  m.impl("dynamic_slice", torch::autograd::autogradNotImplementedFallback());
  m.impl("custom_operation", torch::autograd::autogradNotImplementedFallback());
  m.impl("ctc_beam_search_decoder",
         torch::autograd::autogradNotImplementedFallback());
  m.impl("identity_loss", torch::autograd::autogradNotImplementedFallback());
  m.impl("start_for_loop", torch::autograd::autogradNotImplementedFallback());
  m.impl("end_for_loop", torch::autograd::autogradNotImplementedFallback());
  m.impl("optimizer_group", torch::autograd::autogradNotImplementedFallback());
  m.impl("set_matmul_serialization",
         torch::autograd::autogradNotImplementedFallback());
  m.impl("set_overlap_for_input",
         torch::autograd::autogradNotImplementedFallback());
  m.impl("set_overlap_for_output",
         torch::autograd::autogradNotImplementedFallback());
  m.impl("recomputation_checkpoint",
         torch::autograd::autogradNotImplementedFallback());
  m.impl("set_available_memory",
         torch::autograd::autogradNotImplementedFallback());
  m.impl("begin_multi_conv", torch::autograd::autogradNotImplementedFallback());
  m.impl("end_multi_conv", torch::autograd::autogradNotImplementedFallback());
  m.impl("push_name_scope", torch::autograd::autogradNotImplementedFallback());
  m.impl("pop_name_scope", torch::autograd::autogradNotImplementedFallback());
  m.impl("end_cpu_op", torch::autograd::autogradNotImplementedFallback());
  m.impl("call_cpu_op", torch::autograd::autogradNotImplementedFallback());
  m.impl("set_attribute", torch::autograd::autogradNotImplementedFallback());
  m.impl("clear_attribute", torch::autograd::autogradNotImplementedFallback());
}

// For some reason these operations are first dispatched to AutogradMeta,
// so we ignore and allow them pass through to Meta
TORCH_LIBRARY_IMPL(aten, AutogradMeta, m) {
  m.impl("ctc_loss.Tensor", torch::autograd::autogradNotImplementedFallback());
  m.impl("ctc_loss.IntList", torch::autograd::autogradNotImplementedFallback());
  m.impl("max_pool3d", torch::autograd::autogradNotImplementedFallback());
  m.impl("one_hot", torch::autograd::autogradNotImplementedFallback());
}
TORCH_LIBRARY_IMPL(torch_scatter, AutogradMeta, m) {
  m.impl("scatter_max", torch::autograd::autogradNotImplementedFallback());
  m.impl("scatter_min", torch::autograd::autogradNotImplementedFallback());
}
