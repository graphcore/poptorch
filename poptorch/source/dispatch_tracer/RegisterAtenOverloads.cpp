// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <torch/csrc/jit/ir/ir.h>

#include <string>
#include <unordered_map>

#include "../popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "CommonHelperFunctions.hpp"
#include "poptorch/DispatchTracer.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "dispatchers/JitDispatch.hpp"
#include "dispatchers/MlirDispatch.hpp"
#include "dispatchers/Tracer.hpp"

namespace poptorch {

namespace {
// This is just a useful helper since sometimes we need to pass both keys in.
c10::DispatchKeySet dispatch_key_set{c10::DispatchKey::PrivateUse2,
                                     c10::DispatchKey::AutogradPrivateUse2};

/*
 * The dispatchers are statically registered and called without any additional
 * context so we need a static structure to handle the initial interception.
 * Afterwards we redirect to one of the handlers to avoid keeping around too
 * much static state.
 */
struct GlobalTracerContext {
  // When we are in a live dispatch context. Used to prevent redispatch back
  // to us when we call CPU implementations and to call CPU when we are in
  // BackendSelect and out of scope.
  inline bool isDispatchOn() { return raii_dispatch_context && dispatch_on; }

  // The active dispatcher. Created once upon dispatch start.
  std::unique_ptr<DispatcherBase> active_dispatch;

  // Activates the interceptor which catches the pytorch calls.
  std::unique_ptr<c10::impl::IncludeDispatchKeyGuard> raii_dispatch_context;

  // A simple guard to stop us from redispatching when we are already in a
  // dispatch context.
  bool dispatch_on;
};

GlobalTracerContext context;

/*
 * Small helper to stop us from intercepting CPU calls called by us.
 */
struct DisableDispatchScope {
  // Wrap the existing tracer guard.
  c10::impl::ExcludeDispatchKeyGuard dispatch_intercept_off;

  DisableDispatchScope() : dispatch_intercept_off(dispatch_key_set) {
    // dispatch_intercept_off =  {dispatch_key_set};
    context.dispatch_on = false;
  }

  ~DisableDispatchScope() { context.dispatch_on = true; }
};

at::Tensor &copyInplace(at::Tensor &self, const at::Tensor &src,
                        bool non_blocking) {
  if (!context.isDispatchOn()) {
    return at::native::copy_(self, src, non_blocking);
  }
  // We will also catch pytorch calls called via C++ so we need to disable our
  // dispatch catcher while it is running.
  DisableDispatchScope guard;
  logging::trace("[TRACING-2] Intercepting aten::copy_");
  context.active_dispatch->copyInplace(self, src);

  return self;
}

} // namespace

// Turn on.
void startDispatch() {
  context.raii_dispatch_context =
      std::make_unique<c10::impl::IncludeDispatchKeyGuard>(dispatch_key_set);
  context.dispatch_on = true;
}

// Turn off.
void endDispatch() {
  context.dispatch_on = false;
  context.raii_dispatch_context.reset();
}

// Returns true if the dispatcher is active.
bool isDispatcherActive() {
  // We can't use context.dispatch_on here, because isDispatcherActive() gets
  // used by the handlers, and we disable the dispatcher using
  // DisableDispatchScope when calling handlers.
  return static_cast<bool>(context.active_dispatch);
}

// Take the inputs to the graph and turn them into our IR graph
// inputs/parameters.
void createGraph(TracingMode mode, const std::vector<at::Tensor> &inputs,
                 const std::vector<at::Tensor> &parameters) {
  if (mode == TracingMode::POPART) {
    context.active_dispatch = std::make_unique<JITDispatch>();
  } else if (mode == TracingMode::MLIR) {
// We don't build this on Centos TODO(T49566)
#if POPTORCH_BUILD_MLIR_COMPILER
    context.active_dispatch = std::make_unique<MLIRDispatch>();
#endif
  } else {
    ERROR("Unsupported target");
  }

  context.active_dispatch->createGraph(inputs, parameters);
}

// Special entry point for convolution.
at::Tensor
convolutionKernel(const at::Tensor &input, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias,
                  const at::IntArrayRef stride, const at::IntArrayRef padding,
                  const at::IntArrayRef dilation, const bool transposed,
                  const at::IntArrayRef output_padding, const int64_t groups) {
  // We will also catch pytorch calls called via C++ so we need to disable our
  // dispatch catcher while it is running.
  DisableDispatchScope guard;
  logging::trace("[TRACING-2] Intercepting aten::convolution");

  at::Tensor out = context.active_dispatch->convolution(
      input, weight, bias, stride, padding, dilation, transposed,
      output_padding, groups);

  return out;
}

void fallback(const c10::OperatorHandle &op, c10::Stack *stack) {
  if (!context.isDispatchOn()) {
    // Redirect back to CPU if we are not in our own dispatch context.
    auto &dispatcher = c10::Dispatcher::singleton();
    dispatcher.callBoxed(op, stack);
    return;
  }
  // We will also catch pytorch calls called via C++ so we need to disable our
  // dispatch catcher while it is running.
  DisableDispatchScope guard;

  const c10::FunctionSchema &schema = op.schema();
  logging::trace("[TRACING-2] Intercepting {} ", schema);

  context.active_dispatch->fallback(op, stack);
}

// We don't build this on Centos TODO(T49566)
#if POPTORCH_BUILD_MLIR_COMPILER

std::shared_ptr<MLIRExecutable> compileMLIR() {
  auto *mlir = dynamic_cast<MLIRDispatch *>(context.active_dispatch.get());
  ERROR_ON(mlir == nullptr);
  return mlir->compile();
}

#endif

std::shared_ptr<torch::jit::Graph> getTracedGraph() {
  auto *jit = dynamic_cast<JITDispatch *>(context.active_dispatch.get());
  ERROR_ON_MSG(jit == nullptr, "[User Unreachable] Tracer context is null.");
  auto copied_graph = jit->graph.copy();

  // torch::jit does not copy attributes on the return node, so copy them here
  copied_graph->return_node()->copyAttributes(*jit->graph.return_node());

  // Build a list of nodes marked for deletion.
  std::unordered_set<torch::jit::Node *> to_delete;
  for (torch::jit::Node *node : copied_graph->nodes()) {
    if (isMarkedForDeletion(node)) {
      to_delete.insert(node);
    }
  }

  // Remove the dead nodes.
  searchAndPossiblyDestroy(to_delete);

  return copied_graph;
}

// Record these tensors as being the outputs of the graph.
void markOutputs(const std::vector<at::Tensor> &outputs,
                 const std::vector<at::Tensor> &data_storage,
                 bool output_tuple) {
  // We will also catch pytorch calls called via C++ so we need to disable our
  // dispatch catcher while it is running.
  DisableDispatchScope guard;
  context.active_dispatch->markOutputs(outputs, data_storage, output_tuple);
}

// Appears in 1.10.
#if TORCH_MINOR_VERSION >= 10
// _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device?
// device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat?
// memory_format=None) -> Tensor
at::Tensor toCopy(const at::Tensor &self,
                  c10::optional<at::ScalarType> dtype = c10::nullopt,
                  c10::optional<at::Layout> layout = c10::nullopt,
                  c10::optional<at::Device> device = c10::nullopt,
                  c10::optional<bool> pin = c10::nullopt,
                  bool non_blocking = false,
                  c10::optional<c10::MemoryFormat> fmt = c10::nullopt) {
  if (!context.isDispatchOn()) {
    return at::native::_to_copy(self, dtype, layout, device, pin, non_blocking,
                                fmt);
  }
  // Turn off dispatch so we can call CPU functions without catching them.
  DisableDispatchScope guard;
  logging::trace("[TRACING-2] Intercepting aten::_to_copy");

  at::Tensor out = context.active_dispatch->toCopyInplace(self, dtype, layout,
                                                          device, pin, fmt);
  return out;
}
#endif

at::Tensor
emptyBase(at::IntArrayRef size,
          c10::optional<at::ScalarType> dtype = c10::nullopt,
          c10::optional<at::Layout> layout = c10::nullopt,
          c10::optional<at::Device> device = c10::nullopt,
          c10::optional<bool> pin_memory = c10::nullopt,
          c10::optional<at::MemoryFormat> memory_format = c10::nullopt) {
  // native calls are a dispatch endpoint so will not be redispatched.
  at::Tensor output = at::native::empty_cpu(size, dtype, layout, device,
                                            pin_memory, memory_format);
  // We have to be careful with backend select kernels, we must return the
  // original result if we are not tracing.
  if (!context.isDispatchOn()) {
    return output;
  }
  // Turn off dispatch so we can call CPU functions without catching them.
  DisableDispatchScope guard;
  logging::trace("[TRACING-2] Intercepting empty_base for tensor: {}, {}",
                 output.data_ptr(), toString(output));

  context.active_dispatch->registerEmptyTensor(output);
  return output;
}

at::Tensor emptyMemoryFormat(
    at::IntArrayRef size, c10::optional<at::ScalarType> dtype = c10::nullopt,
    c10::optional<at::Layout> layout = c10::nullopt,
    c10::optional<at::Device> device = c10::nullopt,
    c10::optional<bool> pin_memory = c10::nullopt,
    c10::optional<at::MemoryFormat> memory_format = c10::nullopt) {

  if (!context.isDispatchOn()) {
    return at::native::empty_cpu(size, dtype, layout, device, pin_memory,
                                 memory_format);
  }
  logging::trace("[TRACING-2] Intercepting memory_format");
  return poptorch::emptyBase(size, dtype, layout, device, pin_memory,
                             memory_format);
}
// empty.out(int[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out)
// -> Tensor(a!)
at::Tensor &emptyOut(at::IntArrayRef size, c10::optional<at::MemoryFormat> fmt,
                     at::Tensor &out) {
  if (!context.isDispatchOn()) {
    return at::native::empty_out(size, fmt, out);
  }
  DisableDispatchScope guard;
  logging::trace("[TRACING-2] Intercepting empty.out");

  context.active_dispatch->registerEmptyTensor(out);
  return out;
}

// func: empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None,
// Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor emptyStrided(at::IntArrayRef size, at::IntArrayRef stride,
                        c10::optional<at::ScalarType> dtype = c10::nullopt,
                        c10::optional<at::Layout> layout = c10::nullopt,
                        c10::optional<at::Device> device = c10::nullopt,
                        c10::optional<bool> pin_memory = c10::nullopt) {
  // native calls are a dispatch endpoint so will not be redispatched.
  at::Tensor output = at::native::empty_strided_cpu(size, stride, dtype, layout,
                                                    device, pin_memory);
  // We have to be careful with backend select kernels, we must return the
  // original result if we are not tracing.
  if (!context.isDispatchOn()) {
    return output;
  }
  // Turn off dispatch so we can call CPU functions without catching them.
  DisableDispatchScope guard;
  logging::trace("[TRACING-2] Intercepting empty_strided");

  context.active_dispatch->registerEmptyTensor(output);

  return output;
}

// aten::detach(Tensor(a) self) -> (Tensor(a))
at::Tensor detach(const at::Tensor &self) {
  if (!context.isDispatchOn()) {
    return at::native::detach(self);
  }
  // Turn off dispatch so we can call CPU functions without catching them.
  DisableDispatchScope guard;
  logging::trace("[TRACING-2] Intercepting aten::detach");

  at::Tensor out = context.active_dispatch->detach(self);
  return out;
}

} // namespace poptorch

/*
  The actual dispatcher part. Overriding these keys causes most operations to
  fall through to our fallback catchers.
*/

TORCH_LIBRARY_IMPL(_, PrivateUse2, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
}

TORCH_LIBRARY_IMPL(_, AutogradPrivateUse2, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
}

/*
 The issue appears to be that certain functions are listed in
 native_functions.yml with:
    `
      device_check: NoCheck
      device_guard: False
    `
 I believe these bypass the dispatch fallback mechanism. Therefore to see them
 we have to overload them directly.
*/

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
}

/*
 * We need to override the BackendSelect key as well. This key is used when
 * PyTorch can't work out which backend *should* be dispatched to. We will catch
 * ALL kernels here even when we are not in scope. This means we need to be
 * careful to ensure we are in scope otherwise we should redirect back to
 * PyTorch.
 */
TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {
  // Silence all warnings and info logs. This is due to the backend select
  // warning when we override these kernels.
  auto log_level = FLAGS_caffe2_log_level;
  FLAGS_caffe2_log_level = c10::GLOG_ERROR;

  m.impl("copy_", &poptorch::copyInplace);
  m.impl("empty.memory_format", &poptorch::emptyMemoryFormat);

  m.impl("empty.out", &poptorch::emptyOut);
  m.impl("empty_strided", &poptorch::emptyStrided);

#if TORCH_MINOR_VERSION >= 10
  m.impl("_to_copy", &poptorch::toCopy);
#endif

  // Turn logging back on.
  FLAGS_caffe2_log_level = log_level;
}

/*
 Convolution and convolution backwards kernels are special cases.

 MLIRNpcomp had the same issue.
 https://github.com/llvm/mlir-npcomp/blob/ec611c1e6f44eb5b49c658fd98740000935a1058/frontends/pytorch/csrc/builder/acap_dispatch.cpp#L563

  The code that handles convolution is one gigantic switch statement so it seems
  to be a bit of a special case in the dispatcher. This is the top level of the
  switch.
*/
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse2, m) {
  m.impl("detach", &poptorch::detach);

  m.impl("convolution", &poptorch::convolutionKernel);

  // Once we have our own device target we can target this kernel.
  // m.impl("convolution_overrideable", &poptorch::convolutionKernel);
}

TORCH_LIBRARY_IMPL(poptorch, PrivateUse2, m) {
  m.impl("ipu_print_tensor",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("nop",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("begin_ipu_block",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("end_ipu_block",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("internal_cast",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("custom_operation",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("ctc_beam_search_decoder",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("identity_loss",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("while_loop_begin",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("end_loop_begin",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("start_if_true",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("start_if_false",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("end_if",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("start_for_loop",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("end_for_loop",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("optimizer_group",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("set_matmul_serialization",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("set_overlap_for_input",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("set_overlap_for_output",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("recomputation_checkpoint",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("set_available_memory",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("begin_multi_conv",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("end_multi_conv",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("push_name_scope",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("pop_name_scope",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("begin_autocast",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("suppress_autocast",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("restore_autocast",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("end_cpu_op",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("call_cpu_op",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("set_attribute",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
  m.impl("clear_attribute",
         torch::CppFunction::makeFromBoxedFunction<&poptorch::fallback>());
}
