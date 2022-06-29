// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <ATen/core/List.h>
#include <ATen/native/CPUFallback.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include <set>
#include <string>
#include <unordered_map>

#include "../popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "CommonHelperFunctions.hpp"
#include "Tensor.hpp"
#include "poptorch/DispatchTracer.hpp"
#include "poptorch/InplaceOps.hpp"
#include "poptorch/Utils.hpp"

#include "poptorch_err/ExceptionHandling.hpp"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "dispatchers/IDispatch.hpp"

#if POPTORCH_BUILD_MLIR_COMPILER
#include "dispatchers/JitDispatch.hpp"
#include "dispatchers/MLIRDispatch.hpp"
#endif

namespace poptorch {

namespace {

std::string valueToString(const c10::IValue &ivalue) {
  if (ivalue.isTensor()) {
    return str(ivalue.toTensor());
  }
  // Don't rely on operator<< for everything as we're currently using
  // the XLA dispatch key but using our own Tensor type: bad things
  // might happen if upstream torch tries to print a tensor by itself.
  if (ivalue.isNone() || ivalue.isScalar() || ivalue.isString() ||
      ivalue.isDevice() || ivalue.isStream() || ivalue.isObject() ||
      ivalue.isEnum()) {
    std::stringstream ss;
    ss << ivalue;
    return ss.str();
  }
  if (ivalue.isList()) {
    std::stringstream ss;
    std::string sep;
    ss << ivalue.tagKind() << " [";
    for (const auto &v : ivalue.toList()) {
      ss << sep << valueToString(v);
      sep = ", ";
    }
    ss << "]";
    return ss.str();
  }
  return "<" + ivalue.tagKind() + ">";
}

bool isIpuDevice(const c10::Device &d) {
  // TODO(T59880): replace XLA -> IPU
  return d.type() == c10::DeviceType::XLA;
}

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
  inline bool isDispatchOn() { return dispatch_on; }

  bool hasActiveDispatch() { return static_cast<bool>(_active_dispatch); }

  IDispatch *activeDispatch() {
    ERROR_ON_MSG(!_active_dispatch, "There is no active dispatch");
    return _active_dispatch.get();
  }

  void resetActiveDispatch(std::unique_ptr<IDispatch> new_dispatch) {
    _active_dispatch = std::move(new_dispatch);
  }

  // When setting the location source ignore all the frames containing one of
  // these strings.
  std::vector<std::string> source_location_excludes;

  // A simple guard to stop us from redispatching when we are already in a
  // dispatch context.
  bool dispatch_on{false};

  // A state used to determine if the new tensors we receive from the dispatcher
  // are inputs or parameters.
  // TODO(T61576) Find a better way to identify parameters and buffers.
  bool moving_parameters{false};

  // Return the passed filename if it doesn't match any of the registered
  // exclusions, else an empty c10::optional.
  c10::optional<std::string> filenameIfNotExcluded(const std::string &filename);

  // Each tensor allocated must have a unique id.
  uint64_t next_tensor_id{1};

  // We can't make the difference between inputs and constants so for
  // now we ask the user to manually specify the input tensors.
  // We use TensorImpl* cast as void* to identify them.
  //
  // Note: these should only be used for pointer comparisons and should never
  // be dereferenced as TensorImpl objects as we don't know if they still
  // exist.
  std::set<void *> graph_inputs;

private:
  // The active dispatcher. Created once upon dispatch start.
  std::unique_ptr<IDispatch> _active_dispatch;
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

// Poplar doesn't support long, so cast to int if needed.
// All the downcasts added here must also be handled
// in MLIRExecutor::execute()
at::Tensor downCastIfNeeded(const at::Tensor &t) {
  if (t.scalar_type() == at::ScalarType::Long) {
    return t.to(at::ScalarType::Int);
  }
  if (t.scalar_type() == at::ScalarType::Double) {
    return t.to(at::ScalarType::Float);
  }
  return t;
}

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

// copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
at::Tensor &copyInplace(at::Tensor &self, const at::Tensor &src,
                        bool non_blocking) {
  context.activeDispatch()->setCurrentCodeLocation(
      getPythonInterpreterSourceRange());
  UNUSED(non_blocking);
  logging::trace("[TRACING-2] Intercepting aten::copy_");
  logging::trace("[Input copy_] self {}", str(self));
  logging::trace("[Input copy_] src {}", str(src));

  // TODO(T59880) rename is_xla() -> is_ipu()
  if (self.is_xla()) {
    if (src.is_cpu()) {
      logging::trace("copy_ CPU -> IPU");
      // TODO(T61574) use already allocated self instead of allocating a new
      // tensor.
      if (isParameter(self)) {
        self = context.activeDispatch()->addParameter(downCastIfNeeded(src));
        // Make sure the parameter flag is preserved.
        logging::trace("copy_ parameter CPU -> IPU, new self {}", str(self));
      } else {
        if (context.graph_inputs.count(src.unsafeGetTensorImpl()) > 0) {
          self = context.activeDispatch()->addInput(downCastIfNeeded(src));
        } else {
          self = context.activeDispatch()->addConstant(downCastIfNeeded(src));
        }
        logging::trace("copy_ input CPU -> IPU, new self {}", str(self));
        // Make sure the parameter flag is preserved.
      }
    } else {
      // TODO(T59880) rename is_xla() -> is_ipu()
      ERROR_ON(!src.is_xla());
      logging::trace("copy_ IPU {} -> IPU {}", src.dtype(), self.dtype());
      context.activeDispatch()->copyInplace(self, src);
    }
  } else {
    ERROR_ON(!self.is_cpu());
    // TODO(T59880) rename is_xla() -> is_ipu()
    if (src.is_xla()) {
      logging::trace("copy_ output IPU -> CPU");
      context.activeDispatch()->addOutput(src, self);
    } else {
      ERROR("Unexpected tensor of type "
            << src.unsafeGetTensorImpl()->device_type()
            << ", did you forget to move a tensor to "
               "the IPU?");
    }
  }

  return self;
}

} // namespace

void startParametersMove() { context.moving_parameters = true; }

void endParametersMove() { context.moving_parameters = false; }

// Turn on.
void startDispatch() { context.dispatch_on = true; }

void enableEagerMode() {
#if POPTORCH_BUILD_MLIR_COMPILER
  auto dispatcher = std::make_unique<MLIRDispatch>();
  dispatcher->initCompiler(/*eager_mode =*/true);

  context.resetActiveDispatch(std::move(dispatcher));
#else
  ERROR("PopTorch must be compiled with POPTORCH_BUILD_MLIR_COMPILER=ON to "
        "use eager mode.");
#endif
}

// Turn off.
void endDispatch(bool error_occurred) {
  context.dispatch_on = false;
  if (error_occurred) {
    // If an error occurred we need to destroy the dispatcher as it will be in
    // an inconsistent state.
    destroyDispatcher();
  }
}

void destroyDispatcher() {
// TODO(T49566) We don't build this on Centos
#if POPTORCH_BUILD_MLIR_COMPILER
  if (context.isDispatchOn()) {
    endDispatch();
  }
  context.resetActiveDispatch(nullptr);
#endif
}

void setParameterName(const at::Tensor &tensor, const std::string &name) {
  context.activeDispatch()->setParameterName(tensor, name);
}

std::string getParameterName(torch::jit::Value *value) {
  return context.activeDispatch()->getParameterName(value);
}

// Returns true if the current compilation is being handled using a dispatcher.
//
// This is needed because in some cases, we don't want calls to be dispatched to
// us, but still want to maintain information about the dispatcher.
bool isCompilingWithDispatcher() {
#if POPTORCH_BUILD_MLIR_COMPILER
  return context.hasActiveDispatch();
#else
  return false;
#endif
}

// Returns true if the dispatcher is currently 'on', and should intercept calls
// to us.
bool isDispatcherOn() {
#if POPTORCH_BUILD_MLIR_COMPILER
  return context.isDispatchOn();
#else
  return false;
#endif
}

// Take the inputs to the graph and turn them into our IR graph
// inputs/parameters.
void createGraph(TracingMode mode, const std::vector<at::Tensor> &inputs,
                 const std::vector<std::string> &source_location_excludes) {
  context.source_location_excludes = source_location_excludes;
  if (mode == TracingMode::POPART) {
// TODO(T49566) We don't build this on Centos
#if POPTORCH_BUILD_MLIR_COMPILER
    context.resetActiveDispatch(std::make_unique<JITDispatch>());
#else
    ERROR("PopTorch must be compiled with POPTORCH_BUILD_MLIR_COMPILER=ON to "
          "use the dispatcher");
#endif
  } else if (mode == TracingMode::MLIR) {
// TODO(T49566) We don't build this on Centos
#if POPTORCH_BUILD_MLIR_COMPILER
    context.resetActiveDispatch(std::make_unique<MLIRDispatch>());
#else
    ERROR("PopTorch must be compiled with POPTORCH_BUILD_MLIR_COMPILER=ON to "
          "use the dispatcher");
#endif
  } else {
    ERROR("Unsupported target");
  }

  context.activeDispatch()->setCurrentCodeLocation(
      getPythonInterpreterSourceRange());
  context.activeDispatch()->createGraph();
  context.graph_inputs.clear();
  for (const auto &input : inputs) {
    context.graph_inputs.emplace(
        reinterpret_cast<void *>(input.unsafeGetTensorImpl()));
  }
}

void cpuFallback(const c10::OperatorHandle &op, torch::jit::Stack *stack) {
  const auto name = c10::toString(op.operator_name());

  logging::trace("[CPU Fallback] Running {} on CPU", name);

  // Call the actual boxed CPU fallback.
  at::native::cpu_fallback(op, stack);
}

void fallback(const c10::OperatorHandle &op, c10::Stack *stack) {
  const c10::FunctionSchema &schema = op.schema();
  logging::trace("[TRACING-2] Intercepting {} ", schema);

  context.activeDispatch()->setCurrentCodeLocation(
      getPythonInterpreterSourceRange());
  for (const auto &t : *stack) {
    logging::trace("[Input {}] {}", schema.name(), valueToString(t));
  }
  context.activeDispatch()->fallback(op, stack);
  for (const auto &t : *stack) {
    logging::trace("[Output {}] {}", schema.name(), valueToString(t));
  }
}

// TODO(T49566) We don't build this on Centos
#if POPTORCH_BUILD_MLIR_COMPILER

std::shared_ptr<MLIRExecutor> compileMLIR() {
  auto *mlir = dynamic_cast<MLIRDispatch *>(context.activeDispatch());
  ERROR_ON(mlir == nullptr);
  auto executable = mlir->compile();
  destroyDispatcher();
  return executable;
}

#endif

InplaceGraphInfo getInplaceGraphInfo(size_t num_anchors,
                                     bool replicas_needing_broadcast) {
#if POPTORCH_BUILD_MLIR_COMPILER
  auto *jit = dynamic_cast<JITDispatch *>(context.activeDispatch());
  ERROR_ON_MSG(jit == nullptr, "[User Unreachable] Tracer context is null.");
  return jit->finalizeInplaceGraphInfo(num_anchors, replicas_needing_broadcast);
#else
  UNUSED(num_anchors);
  UNUSED(replicas_needing_broadcast);
  ERROR("PopTorch must be compiled with -DPOPTORCH_BUILD_MLIR_COMPILER=ON");
#endif
}

std::shared_ptr<torch::jit::Graph> getTracedGraph() {
#if POPTORCH_BUILD_MLIR_COMPILER
  auto *jit = dynamic_cast<JITDispatch *>(context.activeDispatch());
  ERROR_ON_MSG(jit == nullptr, "[User Unreachable] Tracer context is null.");

  // Build a list of nodes marked for deletion.
  std::unordered_set<torch::jit::Node *> to_delete;
  for (torch::jit::Node *node : jit->graph->nodes()) {
    if (isMarkedForDeletion(node)) {
      to_delete.insert(node);
    }
  }

  // Remove the dead nodes.
  searchAndPossiblyDestroy(to_delete);

  // Return the real graph because popart_compiler will call
  // getDataSourceForValue() on some of these nodes and if we
  // clone the graph we won't be able to find the mappings.
  return jit->graph;
#else
  ERROR("PopTorch must be compiled with -DPOPTORCH_BUILD_MLIR_COMPILER=ON");
#endif
}

void finalizeGraph() { context.activeDispatch()->finalizeGraph(); }

void *getDataSource(const at::Tensor &tensor) {
  return getCpuData(tensor)->data();
}

void *getDataSourceForValue(torch::jit::Value *value) {
  return context.activeDispatch()->getDataSource(value);
}

bool isParameter(torch::jit::Value *value) {
  return context.activeDispatch()->isParameter(value);
}

// This is the function called by Torch to trigger an IPU to Host
// sync: we forward it to the CPU backend which will then issue
// some copy_ calls between IPU and CPU tensors instead.
at::Scalar localScalarDense(const at::Tensor &self) {
  logging::trace("Sync to CPU");

  return at::native::call_fallback_fn<&poptorch::cpuFallback,
                                      ATEN_OP(_local_scalar_dense)>::call(self);
}

at::Tensor
emptyBase(at::IntArrayRef size,
          c10::optional<at::ScalarType> dtype = c10::nullopt,
          c10::optional<at::Layout> layout = c10::nullopt,
          c10::optional<at::Device> device = c10::nullopt,
          c10::optional<bool> pin_memory = c10::nullopt,
          c10::optional<at::MemoryFormat> memory_format = c10::nullopt) {
  ERROR_ON(!device); // Internal error: shouldn't happen
  if (isIpuDevice(*device)) {
    context.activeDispatch()->setCurrentCodeLocation(
        getPythonInterpreterSourceRange());

    // We use the device ID to determine if a tensor is a parameter
    // (device 1) or not (device 0) but in reality all the tensors
    // currently live on the same IPU so always use the default IPU.
    at::Tensor output = context.activeDispatch()->allocateTensor(
        size, dtype, deviceOrDefaultIpu({}), layout, pin_memory, memory_format);
    // TODO(T61576) Find a better way to identify parameters and buffers.
    setIsParameter(output, context.moving_parameters);

    logging::trace("[TRACING-2] Intercepting IPU empty_base");
    context.activeDispatch()->registerEmptyTensor(output);
    return output;
  }
  // Native calls are a dispatch endpoint so will not be redispatched.
  at::Tensor output = at::native::empty_cpu(size, dtype, layout, device,
                                            pin_memory, memory_format);
  logging::trace("[TRACING-2] Ignoring empty_base for CPU tensor: {}",
                 str(output));
  return output;
}

// Handler for IPU empty tensors: this means the returned tensor must be
// an IPU tensor.
at::Tensor emptyMemoryFormat(
    at::IntArrayRef size, c10::optional<at::ScalarType> dtype = c10::nullopt,
    c10::optional<at::Layout> layout = c10::nullopt,
    c10::optional<at::Device> device = c10::nullopt,
    c10::optional<bool> pin_memory = c10::nullopt,
    c10::optional<at::MemoryFormat> memory_format = c10::nullopt) {

  logging::trace("[TRACING-2] Intercepting IPU memory_format");
  return poptorch::emptyBase(size, dtype, layout, deviceOrDefaultIpu(device),
                             pin_memory, memory_format);
}

// func: empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None,
// Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor emptyStrided(at::IntArrayRef size, at::IntArrayRef stride,
                        c10::optional<at::ScalarType> dtype = c10::nullopt,
                        c10::optional<at::Layout> layout = c10::nullopt,
                        c10::optional<at::Device> device = c10::nullopt,
                        c10::optional<bool> pin_memory = c10::nullopt) {
  ERROR_ON(!device); // Internal error: shouldn't happen
  ERROR_ON(!isIpuDevice(*device));
  logging::trace("[TRACING-2] Intercepting empty_strided");
  ERROR_ON(at::detail::defaultStrides(size) != stride);
  return emptyBase(size, dtype, layout, device, pin_memory);
}

// aten::detach(Tensor(a) self) -> (Tensor(a))
at::Tensor detach(const at::Tensor &self) {
  logging::trace("[TRACING-2] Intercepting aten::detach");

  context.activeDispatch()->setCurrentCodeLocation(
      getPythonInterpreterSourceRange());
  at::Tensor out = context.activeDispatch()->detach(self);
  return out;
}

void replaceValueDispatcher(torch::jit::Value *v_old,
                            torch::jit::Value *v_new) {
  if (!context.hasActiveDispatch()) {
    return;
  }
  context.activeDispatch()->replaceValue(v_old, v_new);
}
} // namespace poptorch

/*
  The actual dispatcher part. Overriding these keys causes most operations to
  fall through to our fallback catchers.
*/

// TODO(T59880) rename XLA -> IPU
TORCH_LIBRARY_IMPL(_, XLA, m) { m.fallback(PTC_BOXED(poptorch::fallback)); }

/* TODO(T59880) Fallback already registered upstream. Re-enable for AutogradIPU
TORCH_LIBRARY_IMPL(_, AutogradIPU, m) {
  m.fallback(PTC_BOXED(poptorch::fallback));
}
*/

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

// TODO(T59880) rename AutogradXLA -> AutogradIPU
// These intercepts are only for ops where we want to override torch's
// autograd behaviour, since the AutogradXLA key has a higher dispatch
// priority than the XLA key. Registration here is not required for
// regular backward ops
TORCH_LIBRARY_IMPL(aten, AutogradXLA, m) {
  m.impl("detach", PTC(poptorch::detach));
}

void popArgumentsFromStack(const c10::OperatorHandle &op, c10::Stack *stack) {
  ERROR_ON(op.schema().arguments().size() > stack->size());
  stack->erase(std::prev(stack->end(), op.schema().arguments().size()),
               stack->end());
}

void pushResultsToStack(c10::Stack *stack,
                        std::vector<c10::IValue> const &results) {
  stack->insert(stack->end(), results.begin(), results.end());
}

// Pop op's arguments from the stack, and (if given) push any results to the
// back.
void updateStack(const c10::OperatorHandle &op, c10::Stack *stack,
                 const std::vector<c10::IValue> &results = {}) {
  popArgumentsFromStack(op, stack);
  if (!results.empty()) {
    pushResultsToStack(stack, results);
  }
}

// Get an argument from the given stack.
c10::IValue getNthArgument(const c10::OperatorHandle &op, c10::Stack *stack,
                           size_t n) {
  ERROR_ON(op.schema().arguments().size() > stack->size());
  return stack->at((stack->size() - op.schema().arguments().size()) + n);
}

void opReturningFirstArgument(const c10::OperatorHandle &op,
                              c10::Stack *stack) {
  if (poptorch::isDispatcherOn()) {
    poptorch::fallback(op, stack);
  } else {
    auto const front = getNthArgument(op, stack, 0);
    updateStack(op, stack, {front});
  }
}

void opWithNoReturn(const c10::OperatorHandle &op, c10::Stack *stack) {
  if (poptorch::isDispatcherOn()) {
    poptorch::fallback(op, stack);
  } else {
    updateStack(op, stack);
  }
}

void callCpuOp(const c10::OperatorHandle &op, c10::Stack *stack) {
  opWithNoReturn(op, stack);

  if (poptorch::isDispatcherOn()) {
    poptorch::endDispatch();
  }
}

void endCpuOp(const c10::OperatorHandle &op, c10::Stack *stack) {
  if (poptorch::isCompilingWithDispatcher()) {
    poptorch::startDispatch();
  }

  opReturningFirstArgument(op, stack);
}

// at::Tensor castOp(at::Tensor tensor, std::string type)
void castOp(const c10::OperatorHandle &op, c10::Stack *stack) {
  if (poptorch::isDispatcherOn()) {
    poptorch::fallback(op, stack);
    return;
  }

  auto type = getNthArgument(op, stack, 1).toString()->string();
  auto tensor = getNthArgument(op, stack, 0).toTensor();

  // If the type to cast to is f16 then we need to cast to f32. The reason being
  // is that by default we will just ignore the type, however this will only
  // work if the original type was f32.

  // Consider:
  /* MyTensor = MyTensor.as(INT8)

    MyTensor = MyTensor.half() # Convert to half.

    out = conv(MyTensor) # This would be an illegal INT8 convolution.
  */
  if (type == "FLOAT16" || type == "FLOAT32") {
    updateStack(op, stack, {tensor.to(at::ScalarType::Float)});
  } else {
    updateStack(op, stack, {tensor});
  }
}

// c10::List<at::Tensor>
// customOperation(c10::List<at::Tensor> inputs,
//                 std::string name, std::string domain,
//                 int64_t version, int64_t num_outputs,
//                 c10::List<at::Tensor> example_outputs,
//                 std::string attributes_map_id) {
//   return example_outputs;
//  }
void customOperation(const c10::OperatorHandle &op, c10::Stack *stack) {
  if (poptorch::isDispatcherOn()) {
    poptorch::fallback(op, stack);
    return;
  }

  auto out = getNthArgument(op, stack, 5);
  updateStack(op, stack, {out});
}

// c10::List<at::Tensor> ctcBeamSearchDecoder(const at::Tensor &log_probs,
//                                            const at::Tensor &lengths,
//                                            int64_t blank, int64_t width,
//                                            int64_t top_paths)
void ctcBeamSearchDecoder(const c10::OperatorHandle &op, c10::Stack *stack) {
  if (poptorch::isDispatcherOn()) {
    poptorch::fallback(op, stack);
    return;
  }

  auto log_probs = getNthArgument(op, stack, 0).toTensor();
  auto top_paths = getNthArgument(op, stack, 3).toInt();
  ERROR_ON_MSG(log_probs.sizes().size() != 3,
               "Incorrect shape for first input to CTC beam search decoder.");
  unsigned input_len = log_probs.sizes()[0];
  unsigned batch_size = log_probs.sizes()[1];

  at::Tensor path_probs = at::zeros({batch_size, top_paths});
  at::Tensor path_lens = at::zeros({batch_size, top_paths});
  at::Tensor decoded_paths = at::zeros({batch_size, top_paths, input_len});

  updateStack(op, stack,
              {c10::List<at::Tensor>({path_probs, path_lens, decoded_paths})});
}

// at::Tensor identityLoss(const at::Tensor &t, int64_t reduction)
void identityLoss(const c10::OperatorHandle &op, c10::Stack *stack) {
  if (poptorch::isDispatcherOn()) {
    poptorch::fallback(op, stack);
    return;
  }

  auto t = getNthArgument(op, stack, 0).toTensor();
  auto reduction = getNthArgument(op, stack, 1).toInt();
  constexpr int64_t sum = 0;
  constexpr int64_t mean = 1;
  constexpr int64_t none = 2;

  popArgumentsFromStack(op, stack);
  switch (reduction) {
  case sum:
    pushResultsToStack(stack, {at::sum(t)});
    return;
  case mean:
    pushResultsToStack(stack, {at::mean(t)});
    return;
  case none:
    pushResultsToStack(stack, {t.clone()});
    return;
  default:
    ERROR("reduction must be sum (0), mean (1) or none (2)");
  }
}

// c10::List<at::Tensor>
// endForLoop(c10::List<at::Tensor> outputs,
//            c10::List<at::Tensor> inputs, int64_t count,
//            c10::List<at::Tensor> example_outputs) {
//   return example_outputs;
// }
void endForLoop(const c10::OperatorHandle &op, c10::Stack *stack) {
  if (poptorch::isDispatcherOn()) {
    poptorch::fallback(op, stack);
    return;
  }

  auto out = getNthArgument(op, stack, 3);
  updateStack(op, stack, {out});
}

// TODO(T64770) This method is the old way of registering custom functions. The
// new way would look like this:
//
// TORCH_LIBRARY(poptorch, m) {
//  m.def("begin_ipu_block(int stage_id, int phase_id, int ipu_id) -> ()",
//        PTC_BOXED(opWithNoReturn));
//  // ...
// }
//
// Unfortunately, with this the trace doesn't pick up on functions that don't
// take a tensor as an input and output a tensor meaning that several of our ops
// don't appear in the traced graph.
static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema("poptorch::begin_ipu_block(int stage_id, int phase_id, "
                        "int ipu_id) -> ()")
                .catchAllKernel<PTC(opWithNoReturn)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::end_ipu_block() -> ()")
                .catchAllKernel<PTC(opWithNoReturn)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::ipu_print_tensor(Tensor self, str? title) "
                        "-> Tensor")
                .catchAllKernel<PTC(opReturningFirstArgument)>())
        .op(torch::RegisterOperators::options()
                .schema(
                    "poptorch::internal_cast(Tensor self, str dtype) -> Tensor")
                .catchAllKernel<PTC(castOp)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::nop(Tensor self) -> Tensor")
                .catchAllKernel<PTC(opReturningFirstArgument)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::custom_operation(Tensor[] inputs, str name, "
                        "str domain, int domain_version, int num_outputs, "
                        "Tensor(a!)[] outputs, str attributes) -> Tensor(a!)[]")
                .catchAllKernel<PTC(customOperation)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::ctc_beam_search_decoder(Tensor probs, "
                        "Tensor lengths, int blank, int beam_width, int "
                        "top_paths) -> Tensor[]")
                .catchAllKernel<PTC(ctcBeamSearchDecoder)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::identity_loss(Tensor x, int reduction) -> "
                        "Tensor")
                .catchAllKernel<PTC(identityLoss)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::start_for_loop(Tensor[] inputs) -> ()")
                .catchAllKernel<PTC(opWithNoReturn)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::end_for_loop(Tensor[] outputs, Tensor[] "
                        "inputs, int trip_count, Tensor(a!)[] example_outputs) "
                        "-> Tensor(a!)[]")
                .catchAllKernel<PTC(endForLoop)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::optimizer_group(int group, Tensor[] inputs) "
                        "-> ()")
                .catchAllKernel<PTC(opWithNoReturn)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::set_matmul_serialization(Tensor matmul, str "
                        "mode, int factor, bool keep_precision) -> Tensor")
                .catchAllKernel<PTC(opReturningFirstArgument)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::set_overlap_for_input(Tensor t, str mode) "
                        "-> Tensor")
                .catchAllKernel<PTC(opReturningFirstArgument)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::set_overlap_for_output(Tensor t, str mode) "
                        "-> Tensor")
                .catchAllKernel<PTC(opReturningFirstArgument)>())
        .op(torch::RegisterOperators::options()
                .schema(
                    "poptorch::recomputation_checkpoint(Tensor self) -> Tensor")
                .catchAllKernel<PTC(opReturningFirstArgument)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::set_available_memory(Tensor t, float mem) "
                        "-> Tensor")
                .catchAllKernel<PTC(opReturningFirstArgument)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::begin_multi_conv() -> ()")
                .catchAllKernel<PTC(opWithNoReturn)>())
        .op(torch::RegisterOperators::options()
                .schema(
                    "poptorch::end_multi_conv(float[]? "
                    "available_memory_proportions, int[]? partials_types, int? "
                    "plan_type, int? per_conv_reserved_tiles, float? "
                    "cycle_back_off, int[]? enableConvDithering) -> ()")
                .catchAllKernel<PTC(opWithNoReturn)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::push_name_scope(str name) -> ()")
                .catchAllKernel<PTC(opWithNoReturn)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::pop_name_scope() -> ()")
                .catchAllKernel<PTC(opWithNoReturn)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::begin_autocast() -> ()")
                .catchAllKernel<PTC(opWithNoReturn)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::suppress_autocast() -> ()")
                .catchAllKernel<PTC(opWithNoReturn)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::restore_autocast() -> ()")
                .catchAllKernel<PTC(opWithNoReturn)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::end_cpu_op(Tensor[] output) -> Tensor[]")
                .catchAllKernel<PTC(endCpuOp)>())
        .op(torch::RegisterOperators::options()
                .schema(
                    "poptorch::call_cpu_op(Tensor[] inputs, str name) -> ()")
                .catchAllKernel<PTC(callCpuOp)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::set_attribute(str attribute, str key, str "
                        "value) -> ()")
                .catchAllKernel<PTC(opWithNoReturn)>())
        .op(torch::RegisterOperators::options()
                .schema(
                    "poptorch::clear_attribute(str attribute, str key) -> ()")
                .catchAllKernel<PTC(opWithNoReturn)>());
