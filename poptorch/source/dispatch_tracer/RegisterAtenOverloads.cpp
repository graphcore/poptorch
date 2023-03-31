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
    activeDispatch()->setPythonStack(torch::jit::tracer::pythonCallstack());
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

  void setPoptorchErrorThrower(PoptorchErrorThrower thrower) {
    _poptorch_error_thrower = std::move(thrower);
  }

private:
  // The active dispatcher. Created once upon dispatch start.
  std::unique_ptr<IDispatch> _active_dispatch;
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
      getContext().tensor_store.copyFromCpu(self, src);
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
        getContext().activeDispatch()->addParameter(downCastIfNeeded(src),
                                                    self);
        // Make sure the parameter flag is preserved.
        ss << "parameter";
      } else {
        ERROR_ON_MSG(
            src.requires_grad(),
            "An input tensor to an IPU model can not have requires_grad set "
            "to True.");

        if (getContext().graph_inputs.count(src.unsafeGetTensorImpl()) > 0) {
          getContext().activeDispatch()->addInput(downCastIfNeeded(src), self);
        } else {
          getContext().activeDispatch()->addConstant(downCastIfNeeded(src),
                                                     self);
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
  ERROR_ON(at::detail::defaultStrides(size) != stride);
  return emptyBase(size, dtype, layout, device, pin_memory);
}

at::Tensor linalgMatrixNorm(const at::Tensor &self, const at::Scalar &ord,
                            at::IntArrayRef dim, bool keepdim,
                            c10::optional<at::ScalarType> dtype) {
  auto ord_double = ord.toDouble();
  auto abs_ord = std::abs(ord_double);
  if (abs_ord != 2.) {
    // As long as we're not dealing with a 2-norm, we can call the
    // operator as usual, which will redispatch the constituent operations
    return at::native::linalg_matrix_norm(self, ord, dim, keepdim, dtype);
  }
  // The 2-norm is defined as the largest (for +2) or smallest (for -2)
  // singular value of the matrix.
  ERROR("Matrix 2-norm is not supported.");
}

at::Tensor linalgMatrixNormStrOrd(const at::Tensor &self, c10::string_view ord,
                                  at::IntArrayRef dim, bool keepdim,
                                  c10::optional<at::ScalarType> dtype) {
  if (ord != "nuc") {
    // As long as we're not dealing with a nuclear norm, we can call the
    // operator as usual, which will redispatch the constituent operations
    return at::native::linalg_matrix_norm(self, ord, dim, keepdim, dtype);
  }
  // The nuclear norm is defined as the sum of singular values of the matrix.
  ERROR("Matrix nuclear norm is not supported.");
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
  RegisterMetaOps.cpp.inc registers the meta implementations of operations
  that are used for type inference
*/
#include "RegisterMetaOps.cpp.inc"
#include "RegisterOptionalAtenOps.cpp.inc"

// These cannot be intercepted using the non-autograd key unless
// torch.inference_mode is used
TORCH_LIBRARY_IMPL(aten, AutogradIPU, m) {
  // This is required to intercept detach calls when moving parameters to the
  // IPU.
  m.impl("detach", PTC_BOXED(poptorch::detach));

  // These must be intercepted at the autograd level otherwise they'll go
  // through fallback
  m.impl("linalg_matrix_norm", PTC(poptorch::linalgMatrixNorm));
  m.impl("linalg_matrix_norm.str_ord", PTC(poptorch::linalgMatrixNormStrOrd));
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
  const auto front = getNthArgument(op, stack, 0);
  updateStack(op, stack, {front});
}

void opWithoutOutputs(const c10::OperatorHandle &op, c10::Stack *stack) {
  if (poptorch::isDispatcherOn()) {
    poptorch::fallback(op, stack);
  } else {
    updateStack(op, stack);
  }
}

void callCpuOp(const c10::OperatorHandle &op, c10::Stack *stack) {
  opWithoutOutputs(op, stack);

  if (poptorch::isDispatcherOn()) {
    poptorch::endDispatch();
  }
}

void endCpuOp(const c10::OperatorHandle &op, c10::Stack *stack) {
  // This op might have been called as part of a CPU model in which case we
  // don't want to re-start the dispatcher.
  if (poptorch::isCompilingWithDispatcher()) {
    poptorch::startDispatch();
    poptorch::fallback(op, stack);
  }

  opReturningFirstArgument(op, stack);
}

at::Tensor castOp(const at::Tensor &tensor, const std::string &type) {
  // If the type to cast to is f16 then we need to cast to f32. The reason being
  // is that by default we will just ignore the type, however this will only
  // work if the original type was f32.

  // Consider:
  /* MyTensor = MyTensor.as(INT8)

    MyTensor = MyTensor.half() # Convert to half.

    out = conv(MyTensor) # This would be an illegal INT8 convolution.
  */
  if (type == "FLOAT16" || type == "FLOAT32") {
    return tensor.to(at::ScalarType::Float);
  }
  return tensor;
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
  auto out = getNthArgument(op, stack, 5);
  updateStack(op, stack, {out});
}

// dynamic_slice(Tensor self, int dim, Tensor start, int size, int step) ->
// Tensor
at::Tensor dynamicSlice(const at::Tensor &self, int64_t dim,
                        const at::Tensor &start, int64_t size, int64_t step) {
  auto st = start.scalar_type();
  std::int64_t start_int;
  if (st == torch::kInt64) {
    start_int = start.data_ptr<std::int64_t>()[0];
  } else if (st == torch::kInt32) {
    start_int = start.data_ptr<std::int32_t>()[0];
  } else if (st == torch::kInt16) {
    start_int = start.data_ptr<std::int16_t>()[0];
  } else {
    ERROR("Expected integer typed start tensor");
  }

  return at::slice(self, dim, {start_int}, {start_int + size}, step);
}

// dynamic_update(Tensor self, Tensor src, int dim, Tensor start, int size) ->
// Tensor
at::Tensor dynamicUpdate(const at::Tensor &self, const at::Tensor &src,
                         int64_t dim, const at::Tensor &start, int64_t size) {
  auto st = start.scalar_type();
  std::int64_t start_int;
  if (st == torch::kInt64) {
    start_int = start.data_ptr<std::int64_t>()[0];
  } else if (st == torch::kInt32) {
    start_int = start.data_ptr<std::int32_t>()[0];
  } else if (st == torch::kInt16) {
    start_int = start.data_ptr<std::int16_t>()[0];
  } else {
    ERROR("Expected integer typed start tensor");
  }

  return at::slice_scatter(self, src, dim, start_int, start_int + size, 1);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
ctcBeamSearchDecoder(const at::Tensor &log_probs,
                     const at::Tensor & /*lengths*/, int64_t /*blank*/,
                     int64_t /*width*/, int64_t top_paths) {
  ERROR_ON_MSG(log_probs.sizes().size() != 3,
               "Incorrect shape for first input to CTC beam search decoder.");
  const unsigned input_len = log_probs.sizes()[0];
  const unsigned batch_size = log_probs.sizes()[1];

  const at::Tensor path_probs = at::zeros({batch_size, top_paths});
  const at::Tensor path_lens = at::zeros({batch_size, top_paths});
  const at::Tensor decoded_paths =
      at::zeros({batch_size, top_paths, input_len});

  return {path_probs, path_lens, decoded_paths};
}

// at::Tensor identityLoss(const at::Tensor &t, int64_t reduction)
at::Tensor identityLoss(const at::Tensor &t, int64_t reduction) {
  constexpr int64_t sum = 0;
  constexpr int64_t mean = 1;
  constexpr int64_t none = 2;

  switch (reduction) {
  case sum:
    return at::sum(t);
  case mean:
    return at::mean(t);
  case none:
    return t.clone();
  default:
    ERROR("reduction must be sum (0), mean (1) or none (2)");
  }
}

#define OP_WITHOUT_OUTPUTS(signature)                                          \
  torch::schema(signature, c10::AliasAnalysisKind::CONSERVATIVE),              \
      PTC_BOXED(opWithoutOutputs)

TORCH_LIBRARY(poptorch, m) {
  // These operations have no outputs, and so are registered with side-effects
  // to prevent being pruned by dead-code elimination
  m.def(OP_WITHOUT_OUTPUTS(
      "begin_ipu_block(int stage_id, int phase_id, int ipu_id) -> ()"));
  m.def(OP_WITHOUT_OUTPUTS("end_ipu_block() -> ()"));
  m.def(OP_WITHOUT_OUTPUTS("start_for_loop(Tensor[] inputs) -> ()"));

  m.def(OP_WITHOUT_OUTPUTS("start_if_block(Tensor condition) -> ()"));
  m.def(OP_WITHOUT_OUTPUTS("start_else_block(Tensor[] outputs_true) -> ()"));

  m.def(
      OP_WITHOUT_OUTPUTS("optimizer_group(int group, Tensor[] inputs) -> ()"));
  m.def(OP_WITHOUT_OUTPUTS("begin_multi_conv() -> ()"));
  m.def(OP_WITHOUT_OUTPUTS(
      "end_multi_conv(float[]? "
      "available_memory_proportions, int[]? partials_types, int? "
      "plan_type, int? per_conv_reserved_tiles, float? "
      "cycle_back_off, int[]? enableConvDithering) -> ()"));
  m.def(OP_WITHOUT_OUTPUTS("push_name_scope(str name) -> ()"));
  m.def(OP_WITHOUT_OUTPUTS("pop_name_scope() -> ()"));
  m.def(OP_WITHOUT_OUTPUTS(
      "set_attribute(str attribute, str key, str value) -> ()"));
  m.def(OP_WITHOUT_OUTPUTS("clear_attribute(str attribute, str key) -> ()"));

  // Operations returning the first argument
  m.def("ipu_print_tensor(Tensor self, str title, int print_gradient, int "
        "summarize_threshold, int edge_items, int max_line_width, int digits, "
        "int float_format, str separator, str open_bracket, str close_bracket) "
        "-> Tensor");
  m.def("nop(Tensor self) -> Tensor");
  m.def("end_for_loop(Tensor[] outputs, Tensor[] "
        "inputs, int trip_count) -> Tensor[]");
  m.def("end_if_block(Tensor[] outputs, Tensor condition) -> Tensor[]");

  m.def("set_matmul_serialization(Tensor matmul, str "
        "mode, int factor, bool keep_precision) -> Tensor");
  m.def("set_overlap_for_input(Tensor t, str mode) -> Tensor");
  m.def("set_overlap_for_output(Tensor t, str mode) -> Tensor");
  m.def("recomputation_checkpoint(Tensor self) -> Tensor");
  m.def("set_available_memory(Tensor t, float mem) -> Tensor");

  m.def("custom_operation(Tensor[] inputs, str name, str domain, int "
        "domain_version, int num_outputs, Tensor[] outputs, str attributes) -> "
        "Tensor[]");
  m.def("ctc_beam_search_decoder(Tensor probs, "
        "Tensor lengths, int blank, int beam_width, int "
        "top_paths) -> (Tensor, Tensor, Tensor)");
  m.def("dynamic_slice(Tensor self, int dim, Tensor start, int size, int step) "
        "-> Tensor");
  m.def("dynamic_update(Tensor self, Tensor src, int dim, Tensor start, int "
        "size) "
        "-> Tensor");
  m.def("identity_loss(Tensor x, int reduction) -> Tensor");
  m.def("internal_cast(Tensor self, str dtype) -> Tensor");

  // call_cpu_op and end_cpu_op are special cases because they must
  // immediately switch the dispatcher on/off so the default poptorch
  // fallback cannot be used. They are also registered with side-effects
  // to ensure they are not reintercepted during constexpr evaluation
  m.def(torch::schema("end_cpu_op(Tensor[] output) -> Tensor[]",
                      c10::AliasAnalysisKind::CONSERVATIVE),
        PTC_BOXED(endCpuOp));
  m.def(torch::schema("call_cpu_op(Tensor[] inputs, str name) -> ()",
                      c10::AliasAnalysisKind::CONSERVATIVE),
        PTC_BOXED(callCpuOp));
}

TORCH_LIBRARY_IMPL(poptorch, CPU, m) {
  // Operations returning the first argument
  m.impl("end_for_loop", PTC_BOXED(opReturningFirstArgument));
  m.impl("end_if_block", PTC_BOXED(opReturningFirstArgument));
  m.impl("ipu_print_tensor", PTC_BOXED(opReturningFirstArgument));
  m.impl("nop", PTC_BOXED(opReturningFirstArgument));
  m.impl("recomputation_checkpoint", PTC_BOXED(opReturningFirstArgument));
  m.impl("set_available_memory", PTC_BOXED(opReturningFirstArgument));
  m.impl("set_matmul_serialization", PTC_BOXED(opReturningFirstArgument));
  m.impl("set_overlap_for_input", PTC_BOXED(opReturningFirstArgument));
  m.impl("set_overlap_for_output", PTC_BOXED(opReturningFirstArgument));

  // Operations with their own CPU implementations
  m.impl("ctc_beam_search_decoder", PTC(ctcBeamSearchDecoder));
  m.impl("custom_operation", PTC_BOXED(customOperation));
  m.impl("dynamic_slice", PTC(dynamicSlice));
  m.impl("dynamic_update", PTC(dynamicUpdate));
  m.impl("identity_loss", PTC(identityLoss));
  m.impl("internal_cast", PTC(castOp));
}

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

  m.impl("start_if_block", torch::autograd::autogradNotImplementedFallback());
  m.impl("start_else_block", torch::autograd::autogradNotImplementedFallback());
  m.impl("end_if_block", torch::autograd::autogradNotImplementedFallback());

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
