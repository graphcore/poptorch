// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include <string>
#include <unordered_map>

#include "../popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "CommonHelperFunctions.hpp"
#include "poptorch/DispatchTracer.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "dispatchers/IDispatch.hpp"

#if POPTORCH_BUILD_MLIR_COMPILER
#include "dispatchers/JitDispatch.hpp"
#include "dispatchers/MlirDispatch.hpp"
#endif

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
  std::unique_ptr<IDispatch> active_dispatch;

  // Activates the interceptor which catches the pytorch calls.
  std::unique_ptr<c10::impl::IncludeDispatchKeyGuard> raii_dispatch_context;

  // When setting the location source ignore all the frames containing one of
  // these strings.
  std::vector<std::string> source_location_excludes;

  // A simple guard to stop us from redispatching when we are already in a
  // dispatch context.
  bool dispatch_on;

  // Return the passed filename if it doesn't match any of the registered
  // exclusions, else an empty c10::optional.
  c10::optional<std::string> filenameIfNotExcluded(const std::string &filename);

  // TODO(T59880) Horrible hack to keep all tensors alive so that the
  // ValueMapper doesn't get confused by storage being re-used by different
  // unrelated tensors.
  std::vector<at::Tensor> tracked_tensors;
};

c10::optional<std::string>
GlobalTracerContext::filenameIfNotExcluded(const std::string &filename) {
  for (auto &exception : source_location_excludes) {
    if (filename.find(exception) != std::string::npos) {
      return {};
    }
  }
  return filename;
}

GlobalTracerContext context;

// adapted from torch/csrc/jit/python/python_tracer.cpp because the header file
// had too many dependepencies
torch::jit::SourceRange getPythonInterpreterSourceRange() {
  auto cs = torch::jit::tracer::pythonCallstack();
  c10::optional<std::string> source_filename;
  size_t source_line = 0;
  std::stringstream stack_trace;
  for (const auto &entry : cs) {
    const auto &range = entry.range;
    if (range.source()) {
      const auto &src = range.source();
      if (src && src->filename()) {
        auto line =
            src->starting_line_no() + src->lineno_for_offset(range.start());
        stack_trace << *(src->filename()) << "(" << line
                    << "): " << entry.filename << "\n";
        if (!source_filename) {
          source_filename = context.filenameIfNotExcluded(*src->filename());
          source_line = line;
        }
      }
    }
  }

  auto stack_trace_text = stack_trace.str();
  auto source = std::make_shared<torch::jit::Source>(
      stack_trace_text, source_filename, source_line);
  if (!source_filename) {
    source_filename = "<unknown>";
  }
  logging::trace("Setting op source to: {}:{}", *source_filename, source_line);
  return torch::jit::SourceRange(source, 0, stack_trace_text.size());
}
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
  context.active_dispatch->setCurrentCodeLocation(
      getPythonInterpreterSourceRange());
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

void destroyDispatcher() {
  if (context.isDispatchOn()) {
    endDispatch();
  }
  context.active_dispatch.reset();
}

// Take the inputs to the graph and turn them into our IR graph
// inputs/parameters.
void createGraph(TracingMode mode, const std::vector<at::Tensor> &inputs,
                 const std::vector<at::Tensor> &parameters,
                 const std::vector<std::string> &source_location_excludes) {
  context.source_location_excludes = source_location_excludes;
  if (mode == TracingMode::POPART) {
// We don't build this on Centos TODO(T49566)
#if POPTORCH_BUILD_MLIR_COMPILER
    context.active_dispatch = std::make_unique<JITDispatch>();
#else
    ERROR("PopTorch must be compiled with POPTORCH_BUILD_MLIR_COMPILER=ON to "
          "use the dispatcher");
#endif
  } else if (mode == TracingMode::MLIR) {
// We don't build this on Centos TODO(T49566)
#if POPTORCH_BUILD_MLIR_COMPILER
    context.active_dispatch = std::make_unique<MLIRDispatch>();
#else
    ERROR("PopTorch must be compiled with POPTORCH_BUILD_MLIR_COMPILER=ON to "
          "use the dispatcher");
#endif
  } else {
    ERROR("Unsupported target");
  }

  context.active_dispatch->setCurrentCodeLocation(
      getPythonInterpreterSourceRange());
  context.active_dispatch->createGraph();
  // TODO(T59880) Adding tensors to tracked_tensor is a horrible hack to keep
  // all tensors alive so that the ValueMapper doesn't get confused by storage
  // being re-used by different unrelated tensors.
  for (const auto &input : inputs) {
    context.tracked_tensors.push_back(context.active_dispatch->addInput(input));
  }
  for (const auto &param : parameters) {
    context.tracked_tensors.push_back(
        context.active_dispatch->addParameter(param));
  }
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

  context.active_dispatch->setCurrentCodeLocation(
      getPythonInterpreterSourceRange());
  context.active_dispatch->fallback(op, stack);
}

// We don't build this on Centos TODO(T49566)
#if POPTORCH_BUILD_MLIR_COMPILER

std::shared_ptr<MLIRExecutable> compileMLIR() {
  auto *mlir = dynamic_cast<MLIRDispatch *>(context.active_dispatch.get());
  ERROR_ON(mlir == nullptr);
  auto executable = mlir->compile();
  destroyDispatcher();
  return executable;
}

#endif

std::shared_ptr<torch::jit::Graph> getTracedGraph() {
#if POPTORCH_BUILD_MLIR_COMPILER
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
#else
  ERROR("PopTorch must be compiled with -DPOPTORCH_BUILD_MLIR_COMPILER=ON");
#endif
}

// Record these tensors as being the outputs of the graph.
void markOutputs(const std::vector<at::Tensor> &outputs,
                 const std::vector<at::Tensor> &data_storage) {
  // We will also catch pytorch calls called via C++ so we need to disable our
  // dispatch catcher while it is running.
  DisableDispatchScope guard;
  context.active_dispatch->setCurrentCodeLocation(
      getPythonInterpreterSourceRange());
  ERROR_ON(outputs.size() != data_storage.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    context.active_dispatch->addOutput(outputs.at(i), data_storage.at(i));
  }
  context.active_dispatch->finalizeGraph();
}

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

  context.active_dispatch->setCurrentCodeLocation(
      getPythonInterpreterSourceRange());
  at::Tensor out = context.active_dispatch->toCopyInplace(self, dtype, layout,
                                                          device, pin, fmt);
  return out;
}

at::Tensor
emptyBase(at::IntArrayRef size,
          c10::optional<at::ScalarType> dtype = c10::nullopt,
          c10::optional<at::Layout> layout = c10::nullopt,
          c10::optional<at::Device> device = c10::nullopt,
          c10::optional<bool> pin_memory = c10::nullopt,
          c10::optional<at::MemoryFormat> memory_format = c10::nullopt) {
  // Native calls are a dispatch endpoint so will not be redispatched.
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
                 static_cast<void *>(output.unsafeGetTensorImpl()),
                 toString(output));

  context.active_dispatch->setCurrentCodeLocation(
      getPythonInterpreterSourceRange());
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

  context.active_dispatch->setCurrentCodeLocation(
      getPythonInterpreterSourceRange());
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

  context.active_dispatch->setCurrentCodeLocation(
      getPythonInterpreterSourceRange());
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

  context.active_dispatch->setCurrentCodeLocation(
      getPythonInterpreterSourceRange());
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
  There are two kinds of PyTorch ops: the ones that require registration
  (and a backend-specific kernel) and the ones that are optional. If optional
  ops are not registered they get decomposed into several required ops that must
  then be intercepted by the backend provider. More information on this can be
  found at https://pytorch.org/tutorials/advanced/extend_dispatcher.html.

  In essence:
    - required ops have 'dispatch' set to TRUE and 'default' set to FALSE in
      RegistrationDeclarations.h
    - optional ops have 'dispatch' set to FALSE or 'default' set to TRUE in
      RegistrationDeclarations.h

  RegisterOptionalAtenOps.cpp.inc registers the optional ops that our backend
  intercepts.
*/
#include "RegisterOptionalAtenOps.cpp.inc"

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
  m.impl("_to_copy", &poptorch::toCopy);

  // Turn logging back on.
  FLAGS_caffe2_log_level = log_level;
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse2, m) {
  m.impl("detach", &poptorch::detach);
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
