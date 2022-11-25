// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <ATen/Operators.h>
#include <ATen/core/List.h>
#include <ATen/core/function_schema.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorImpl.h>
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/types.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

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
#include "pytorch_bridge/IpuSession.hpp"

#include "dispatchers/JitDispatch.hpp"

#include "pytorch_bridge/CompilerOptions.hpp"

// The functions in this file are called via Torch's dispatcher, therefore
// we should only catch the exceptions which are not handled by
// the dispatcher.
#define PTC(f)                                                                 \
  PoptorchCatchWrapperImpl<poptorch::throwPoptorchError, /*catch_all=*/false,  \
                           decltype(&(f)), f>::wrap
#define PTC_BOXED(f) torch::CppFunction::makeFromBoxedFunction<PTC(f)>()

namespace poptorch {

namespace {

std::string valueToString(const c10::IValue &ivalue) {
  if (ivalue.isTensor()) {
    return str(ivalue.toTensor());
  }
  // TODO(T59880)
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
  return d.type() == c10::DeviceType::IPU;
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

  void updatePythonCallstack() {
    if (_python_traceback_accessor) {
      activeDispatch()->setPythonStack(_python_traceback_accessor());
    } else {
      activeDispatch()->setPythonStack({});
    }
  }

  void throwPoptorchError(const PoptorchErrorInfo &info) {
    if (_poptorch_error_thrower) {
      _poptorch_error_thrower(info);
    }
  }

  // A simple guard to stop us from redispatching when we are already in a
  // dispatch context.
  bool dispatch_on{false};

  // A state used to determine if the new tensors we receive from the dispatcher
  // are inputs or parameters.
  bool moving_parameters{false};

  // A state used to determine whether we are currently registering output
  // tensors for the graph (in IPUScope.outputs()). If we're not, moving
  // output tensors may result in bad data, so we warn. An example of when
  // this might happen is using torch dynamic slicing in the dispatcher
  // (instead of poptorch.dynamic_slice()).
  bool moving_outputs{false};

  // We can't make the difference between inputs and constants so for
  // now we ask the user to manually specify the input tensors.
  // We use TensorImpl* cast as void* to identify them.
  //
  // Note: these should only be used for pointer comparisons and should never
  // be dereferenced as TensorImpl objects as we don't know if they still
  // exist.
  std::set<void *> graph_inputs;

  // Create and store Tensors...
  TensorStore tensor_store;

  void setPythonTracebackAccessor(PythonTracebackAccessor accessor) {
    _python_traceback_accessor = std::move(accessor);
  }

  void setPoptorchErrorThrower(PoptorchErrorThrower thrower) {
    _poptorch_error_thrower = std::move(thrower);
  }

private:
  // The active dispatcher. Created once upon dispatch start.
  std::unique_ptr<IDispatch> _active_dispatch;
  PythonTracebackAccessor _python_traceback_accessor;
  PoptorchErrorThrower _poptorch_error_thrower;
};

std::unique_ptr<GlobalTracerContext> context =
    std::make_unique<GlobalTracerContext>();

GlobalTracerContext &getContext() { return *context; }

// Poplar doesn't support long, so cast to int if needed.
at::Tensor downCastIfNeeded(const at::Tensor &t) {
  if (t.scalar_type() == at::ScalarType::Long) {
    return t.to(at::ScalarType::Int);
  }
  if (t.scalar_type() == at::ScalarType::Double) {
    return t.to(at::ScalarType::Float);
  }
  return t;
}

// NOLINTNEXTLINE
void hostSideCast(void *dest, c10::ScalarType dest_scalar_type, void *src,
                  const void *src_end, c10::ScalarType src_scalar_type) {
  // NOLINTNEXTLINE
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, dest_scalar_type, "copy_", [&] {
        using dest_t = scalar_t;

        // NOLINTNEXTLINE
        AT_DISPATCH_ALL_TYPES_AND(
            at::ScalarType::Half, src_scalar_type, "copy_", [&] {
              scalar_t *src_ = reinterpret_cast<scalar_t *>(src);
              dest_t *dest_ = reinterpret_cast<dest_t *>(dest);

              // TODO(T69558): use vectorised casts
              // at::vec::convert(src, dest, numel);

              while (reinterpret_cast<void *>(src_) != src_end) {
                *(dest_++) =
                    c10::static_cast_with_inter_type<dest_t, scalar_t>::apply(
                        *(src_++));
              }
            });
      });
}

// Return true if the given IPU tensor is a parameter.
inline bool isParameter(const at::Tensor &tensor) {
  ERROR_ON(!getContext().hasActiveDispatch());
  return getContext().activeDispatch()->isParameter(tensor);
}

// copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
void copyInplace(const c10::OperatorHandle &op, c10::Stack *stack) {
  const c10::FunctionSchema &schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  auto arguments = torch::jit::last(stack, num_arguments);

  // In an ideal world self would be allowed to change to reflect type coercion.
  // Unfortunately, pytorch's boxed function interface does not properly support
  // outputs. To work around this if need to re-allocate self we map both the
  // new and old values is the value mapper within the dispatcher.
  // Self is marked as const here to ensure we don't accidentally change it
  const at::Tensor self = arguments.at(0).toTensor();
  const at::Tensor src = arguments.at(1).toTensor();

  logging::debug("[DISPATCHER] Intercepting aten::copy_");
  logging::trace("[Input] self {}", str(self));
  logging::trace("[Input] src {}", str(src));

  // In eager mode the dispatcher is always active so this will only be true
  // when working with static graphs
  if (!getContext().hasActiveDispatch()) {
    if (self.is_ipu() && src.is_cpu()) {
      logging::trace("copy_ CPU -> IPU, outside dispatch");
      auto scalar_type = src.scalar_type();
      auto coerced_type = coerceToSupportedType(scalar_type);
      ERROR_ON_MSG(scalar_type != coerced_type,
                   "Unsupported scalar type `"
                       << scalar_type << "'. Please cast to `" << coerced_type
                       << "' before moving this tensor to the IPU.");
      // TODO(T71349): contiguous is needed to ensure the cpu data has the
      // standard striding layout and is continuous in memory
      getContext().tensor_store.copyFromCpu(self, src.contiguous());
    } else if (self.is_cpu() && src.is_ipu()) {
      logging::trace("copy_ IPU -> CPU, outside dispatch");
      getContext().tensor_store.copyToCpu(self, src);
    } else if (self.is_ipu() && src.is_ipu()) {
      if (!getHostBuffer(self).hasData()) {
        getContext().tensor_store.allocateBuffer(self);
      }

      const auto &self_buffer = getHostBuffer(self).getCpuData();
      const auto &src_buffer = getHostBuffer(src).getCpuData();

      ERROR_ON(!src_buffer);

      if (self.dtype() != src.dtype()) {
        logging::trace("copy_ cast from {} to {} on CPU, outside dispatch",
                       src.dtype(), self.dtype());
        hostSideCast(
            self_buffer->data(), self.scalar_type(), src_buffer->data(),
            src_buffer->data() + src_buffer->size(), src.scalar_type());
      } else {
        ERROR_ON_MSG(self_buffer->size() != src_buffer->size(),
                     "Failed to copy_ outside dispatch: src and self host-side "
                     "buffer sizes are not equal.");
        *self_buffer = *src_buffer;
      }
    } else {
      ERROR("Intercepted unexpected copy_ outside dispatch: only copies "
            "between CPU, IPU tensors as well as between IPU tensors "
            "themselves are supported.");
    }

    torch::jit::drop(stack, num_arguments);
    torch::jit::push(stack, self);
    return;
  }

  getContext().updatePythonCallstack();

  if (self.is_ipu()) {
    if (src.is_cpu()) {
      std::stringstream ss;
      ss << "copy_ CPU -> IPU ";
      if (isParameter(self) || getContext().moving_parameters) {
        // TODO(T71349): src.contiguous() is needed to ensure the cpu data has
        // the standard striding layout and is continuous in memory
        getContext().activeDispatch()->addParameter(
            downCastIfNeeded(src.contiguous()), self);
        // Make sure the parameter flag is preserved.
        ss << "parameter";
      } else {
        ERROR_ON_MSG(
            src.requires_grad(),
            "An input tensor to an IPU model can not have requires_grad set "
            "to True.");

        // TODO(T71349): src.contiguous() is needed to ensure the cpu data has
        // the standard striding layout and is continuous in memory
        if (getContext().graph_inputs.count(src.unsafeGetTensorImpl()) > 0) {
          getContext().activeDispatch()->addInput(
              downCastIfNeeded(src.contiguous()), self);
        } else {
          getContext().activeDispatch()->addConstant(
              downCastIfNeeded(src.contiguous()), self);
        }
        ss << "input";
        // Make sure the parameter flag is preserved.
      }
      ss << ", new self " << str(self);
      logging::debug(ss.str().c_str());

      torch::jit::drop(stack, num_arguments);
      torch::jit::push(stack, self);
    } else {
      ERROR_ON(!src.is_ipu());
      logging::debug("copy_ IPU {} -> IPU {}", src.dtype(), self.dtype());
      getContext().activeDispatch()->fallback(op, stack);
    }
  } else {
    ERROR_ON(!self.is_cpu());
    if (src.is_ipu()) {
      ERROR_ON_MSG(!getContext().moving_outputs,
                   "Illegal move to CPU (via `.to(\"cpu\")`) when using the "
                   "dispatcher. Instead, return this output as an IPU tensor.");
      logging::debug("copy_ output IPU -> CPU");
      getContext().activeDispatch()->addOutput(src, self);

      torch::jit::drop(stack, num_arguments);
      torch::jit::push(stack, self);
    } else {
      ERROR("Unexpected tensor of type "
            << src.unsafeGetTensorImpl()->device_type()
            << ", did you forget to move a tensor to "
               "the IPU?");
    }
  }
}

} // namespace

void startParametersMove() { getContext().moving_parameters = true; }

void endParametersMove() { getContext().moving_parameters = false; }

void startOutputsMove() { getContext().moving_outputs = true; }

void endOutputsMove() { getContext().moving_outputs = false; }

// Turn on.
void startDispatch() { getContext().dispatch_on = true; }

void setPythonTracebackAccessor(PythonTracebackAccessor accessor) {
  getContext().setPythonTracebackAccessor(std::move(accessor));
}

void setPoptorchErrorThrower(PoptorchErrorThrower thrower) {
  getContext().setPoptorchErrorThrower(std::move(thrower));
}

void throwPoptorchError(const PoptorchErrorInfo &info) {
  getContext().throwPoptorchError(info);
}

// Turn off.
void endDispatch(bool error_occurred) {
  getContext().dispatch_on = false;
  if (error_occurred) {
    // If an error occurred we need to destroy the dispatcher as it will be in
    // an inconsistent state.
    destroyDispatcher();
  }
}

// Cleanup on exit callback to avoid global destructor ordering issues
void poptorchAtExit() {
  // Ensure that the context is deleted before globals are destroyed to avoid
  // issues with global destructor ordering
  context.reset();
}

// Destroys the dispatcher after we have finished compiling
void destroyDispatcher() {
  if (getContext().isDispatchOn()) {
    endDispatch();
  }
  getContext().resetActiveDispatch(nullptr);
}

void setParameterName(const at::Tensor &tensor, const std::string &name) {
  getContext().activeDispatch()->setParameterName(tensor, name);
}

std::string getParameterName(torch::jit::Value *value) {
  return getContext().activeDispatch()->getParameterName(value);
}

void setParameterPerReplica(const std::string &param_name,
                            const at::Tensor &tensor, int comm_group_type,
                            int shards, int variable_retrieval_mode) {
  getContext().activeDispatch()->setParameterPerReplica(
      param_name, tensor, comm_group_type, shards, variable_retrieval_mode);
}

bool getParameterPerReplica(torch::jit::Value *value,
                            PerReplicaSettings &settings) {
  return getContext().activeDispatch()->getParameterPerReplica(value, settings);
}
// Returns true if the current compilation is being handled using a dispatcher.
//
// This is needed because in some cases, we don't want calls to be dispatched to
// us, but still want to maintain information about the dispatcher.
bool isCompilingWithDispatcher() { return getContext().hasActiveDispatch(); }

// Returns true if the dispatcher is currently 'on', and should intercept calls
// to us.
bool isDispatcherOn() { return getContext().isDispatchOn(); }

CompilerOptions
createMLIROptions(const std::vector<std::string> &source_location_excludes) {
  CompilerOptions options;
  std::transform(
      source_location_excludes.begin(), source_location_excludes.end(),
      std::back_inserter(options.dispatcher.source_location_excludes),
      [](const std::string &exclude) {
        return std::vector<char>(exclude.begin(), exclude.end());
      });
  return options;
}

// Take the inputs to the graph and turn them into our IR graph
// inputs/parameters.
void createGraph(TracingMode mode, const std::vector<at::Tensor> &inputs,
                 const CompilerOptions &options) {
  if (mode == TracingMode::POPART) {
    getContext().resetActiveDispatch(
        std::make_unique<JITDispatch>(options, &getContext().tensor_store));
  } else {
    ERROR("Unsupported target");
  }

  getContext().updatePythonCallstack();
  getContext().graph_inputs.clear();
  for (const auto &input : inputs) {
    getContext().graph_inputs.emplace(
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
  logging::debug("[DISPATCHER] Intercepting {} ", schema);

  getContext().updatePythonCallstack();
  for (const auto &t : *stack) {
    logging::trace("[Input {}] {}", schema.name(), valueToString(t));
  }
  getContext().activeDispatch()->fallback(op, stack);
  for (const auto &t : *stack) {
    logging::trace("[Output {}] {}", schema.name(), valueToString(t));
  }
}

InplaceGraphInfo getInplaceGraphInfo(size_t num_anchors,
                                     bool replicas_needing_broadcast) {
  auto *jit = dynamic_cast<JITDispatch *>(getContext().activeDispatch());
  ERROR_ON_MSG(jit == nullptr, "[User Unreachable] Tracer context is null.");
  return jit->finalizeInplaceGraphInfo(num_anchors, replicas_needing_broadcast);
}

std::shared_ptr<torch::jit::Graph> getTracedGraph() {
  auto *jit = dynamic_cast<JITDispatch *>(getContext().activeDispatch());
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
}

void finalizeGraph() { getContext().activeDispatch()->finalizeGraph(); }

void *getDataSource(const at::Tensor &tensor) {
  return getHostBuffer(tensor).getCpuData()->data();
}

void *getDataSourceForValue(torch::jit::Value *value) {
  return getContext().activeDispatch()->getDataSource(value);
}

bool isParameter(torch::jit::Value *value) {
  return getContext().activeDispatch()->isParameter(value);
}

// This is the function called by Torch to trigger an IPU to Host
// sync: we forward it to the CPU backend which will then issue
// some copy_ calls between IPU and CPU tensors instead.
at::Scalar localScalarDense(const at::Tensor &self) {
  logging::trace("Sync to CPU");

  return at::native::call_fallback_fn<&poptorch::cpuFallback,
                                      ATEN_OP(_local_scalar_dense)>::call(self);
}

at::Scalar item(const at::Tensor &self) {
  ERROR("aten::item is only supported in eager mode, but was intercepted in "
        "a static graph. This means an IPU to CPU copy was triggered before "
        "the end of the graph, for example by calling tensor.item(). "
        "Please ensure that any such copies are removed.");

  return at::native::call_fallback_fn<&poptorch::cpuFallback,
                                      ATEN_OP(item)>::call(self);
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
    // We use the device ID to determine if a tensor is a parameter
    // (device 1) or not (device 0) but in reality all the tensors
    // currently live on the same IPU so always use the default IPU.
    at::Tensor output = getContext().tensor_store.allocateTensor(
        size, dtype, nullptr, deviceOrDefaultIpu({}));
    // TODO(T61576) Find a better way to identify parameters and buffers.
    if (getContext().hasActiveDispatch()) {
      getContext().updatePythonCallstack();
      getContext().activeDispatch()->registerEmptyTensor(
          output, getContext().moving_parameters);
    }

    return output;
  }
  // Native calls are a dispatch endpoint so will not be redispatched.
  at::Tensor output = at::native::empty_cpu(size, dtype, layout, device,
                                            pin_memory, memory_format);
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

  auto device_or_default = deviceOrDefaultIpu(device);
  logging::debug(
      "[DISPATCHER] Intercepting aten::empty.memory_format, device {}",
      device_or_default.str());
  return poptorch::emptyBase(size, dtype, layout, device_or_default, pin_memory,
                             memory_format);
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
  logging::debug("[DISPATCHER] Intercepting aten::empty_strided, device {}",
                 device->str());
  // TODO(T71349): once upstream torch understands that the ipu tensor doesn't
  // support striding the striding should always be default
  // ERROR_ON(at::detail::defaultStrides(size) != stride);
  UNUSED(stride);
  return emptyBase(size, dtype, layout, device, pin_memory);
}

// aten::detach(Tensor(a) self) -> (Tensor(a))
void detach(const c10::OperatorHandle &op, c10::Stack *stack) {
  logging::debug("[DISPATCHER] Intercepting aten::detach");

  if (getContext().hasActiveDispatch()) {
    getContext().updatePythonCallstack();

    // Perform the shallow copy and detach.
    getContext().activeDispatch()->detach(op, stack,
                                          getContext().moving_parameters);
  } else {
    const c10::FunctionSchema &schema = op.schema();
    const auto num_arguments = schema.arguments().size();
    const auto arguments = torch::jit::last(stack, num_arguments);

    ERROR_ON(arguments.size() != 1);
    const at::Tensor in = arguments.front().toTensor();

    const at::Tensor out(in.unsafeGetTensorImpl()->shallow_copy_and_detach(
        /*version_counter=*/in.unsafeGetTensorImpl()->version_counter(),
        /*allow_tensor_metadata_change=*/true));

    torch::jit::drop(stack, num_arguments);
    torch::jit::push(stack, out);
  }
}

// NOTE: This gets called by _weight_norm's handler, if certain conditions are
// met. However, those conditions never used to be met, and so we never had to
// implement this handler. Now we do, so for now just emulate the old behaviour.
void weightNormInterface(const c10::OperatorHandle &op, c10::Stack *stack) {
  const auto num_arguments = op.schema().arguments().size();
  auto arguments = torch::jit::last(stack, num_arguments);

  const auto v = arguments.at(0).toTensor();
  const auto g = arguments.at(1).toTensor();
  const std::int64_t dim = arguments.at(2).toInt();

  torch::jit::drop(stack, num_arguments);

  const auto out = v * (g / at::norm_except_dim(v, 2, dim));

  torch::jit::push(stack, out);
  // Strictly speaking the schema of `_weight_norm_interface` returns a
  // (Tensor, Tensor); in its sole usage in `_weight_norm`, only the first
  // member is used, so just return something empty of the right shape.
  torch::jit::push(stack, at::empty_like(g));
}

void replaceValueDispatcher(torch::jit::Value *v_old,
                            torch::jit::Value *v_new) {
  if (!getContext().hasActiveDispatch()) {
    return;
  }
  getContext().activeDispatch()->replaceValue(v_old, v_new);
}

std::uint64_t getIpuTensorId(const at::Tensor &tensor) {
  ERROR_ON_MSG(!isIpuTensor(tensor),
               "You may only call getIpuTensorId on an IPU tensor");
  return ipuTensorId(tensor);
}

} // namespace poptorch

/*
  The actual dispatcher part. Overriding these keys causes most operations to
  fall through to our fallback catchers.
*/

TORCH_LIBRARY_IMPL(_, IPU, m) { m.fallback(PTC_BOXED(poptorch::fallback)); }

TORCH_LIBRARY_IMPL(_, AutogradIPU, m) {
  m.fallback(PTC_BOXED(poptorch::fallback));
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

// This is required to intercept detach calls when
// moving parameters to the IPU. For some reason, these
// cannot be intercepted using the non-autograd key
TORCH_LIBRARY_IMPL(aten, AutogradIPU, m) {
  m.impl("detach", PTC_BOXED(poptorch::detach));
}

void popArgumentsFromStack(const c10::OperatorHandle &op, c10::Stack *stack) {
  ERROR_ON(op.schema().arguments().size() > stack->size());
  stack->erase(std::prev(stack->end(), op.schema().arguments().size()),
               stack->end());
}

void pushResultsToStack(c10::Stack *stack,
                        const std::vector<c10::IValue> &results) {
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
    const auto front = getNthArgument(op, stack, 0);
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
  // This op might have been called as part of a CPU model in which case we
  // don't want to re-start the dispatcher.
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
    // NB treated as inplace due to use of example_outputs
    poptorch::fallback(op, stack);
    return;
  }

  auto out = getNthArgument(op, stack, 5);
  updateStack(op, stack, {out});
}

// dynamic_slice(Tensor self, int dim, Tensor start, int size, int step) ->
// Tensor
void dynamicSlice(const c10::OperatorHandle &op, c10::Stack *stack) {
  if (poptorch::isDispatcherOn()) {
    poptorch::fallback(op, stack);
  } else {
    const auto self = getNthArgument(op, stack, 0).toTensor();
    auto dim = getNthArgument(op, stack, 1).toInt();
    auto t_start = getNthArgument(op, stack, 2).toTensor();
    auto st = t_start.scalar_type();
    std::int64_t start;
    if (st == torch::kInt64) {
      start = t_start.data_ptr<std::int64_t>()[0];
    } else if (st == torch::kInt32) {
      start = t_start.data_ptr<std::int32_t>()[0];
    } else if (st == torch::kInt16) {
      start = t_start.data_ptr<std::int16_t>()[0];
    } else {
      ERROR("Expected integer typed start tensor");
    }

    auto size = getNthArgument(op, stack, 3).toInt();
    auto step = getNthArgument(op, stack, 4).toInt();

    auto result = at::slice(self, dim, {start}, {start + size}, step);

    updateStack(op, stack, {result});
  }
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
  const unsigned input_len = log_probs.sizes()[0];
  const unsigned batch_size = log_probs.sizes()[1];

  const at::Tensor path_probs = at::zeros({batch_size, top_paths});
  const at::Tensor path_lens = at::zeros({batch_size, top_paths});
  const at::Tensor decoded_paths =
      at::zeros({batch_size, top_paths, input_len});

  updateStack(op, stack, {path_probs, path_lens, decoded_paths});
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
                .schema("poptorch::dynamic_slice(Tensor self, int dim, "
                        "Tensor start, int size, int step) -> Tensor")
                .catchAllKernel<PTC(dynamicSlice)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::custom_operation(Tensor[] inputs, str name, "
                        "str domain, int domain_version, int num_outputs, "
                        "Tensor[] outputs, str attributes) -> Tensor[]")
                .catchAllKernel<PTC(customOperation)>())
        .op(torch::RegisterOperators::options()
                .schema("poptorch::ctc_beam_search_decoder(Tensor probs, "
                        "Tensor lengths, int blank, int beam_width, int "
                        "top_paths) -> (Tensor, Tensor, Tensor)")
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
                        "inputs, int trip_count) -> Tensor[]")
                .catchAllKernel<PTC(opReturningFirstArgument)>())
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

// By default, if we don't register anything for autograd, the the outputs of
// `poptorch::` ops will have no `grad_fn` (making them leaves). For PopART it's
// not inherently an issue since PopART does its own thing in the backward pass.
// However, PyTorch will error if you put the output of one of these ops through
// an inplace op: `a leaf Variable that requires grad is being used in an
// in-place operation.`
//
// The JIT trace will have the `grad_fn`s filled with whatever the previous
// `grad_fn` of the input was, so this isn't an issue.
//
// Note: Presumably, for non-PopART backends these will need to have
// implementations (`torch::autograd::Function` subclasses).
TORCH_LIBRARY_IMPL(poptorch, AutogradIPU, m) {
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
