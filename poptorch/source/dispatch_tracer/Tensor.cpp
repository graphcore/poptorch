// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "Tensor.hpp"

#include <ATen/ATen.h>
#include <ATen/OpaqueTensorImpl.h>
#include <c10/core/ScalarType.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <variant>
#include <vector>

#include "CommonHelperFunctions.hpp"
#include "ValueMapper.hpp"

#include "poptorch/DispatchTracer.hpp"

#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"
#include "pytorch_bridge/CompilerTypes.hpp"
#include "pytorch_bridge/IpuSession.hpp"

namespace poptorch {

namespace {

using BufferPtr = std::shared_ptr<Buffer>;
using TensorViewPtr = std::shared_ptr<ITensorView>;

template <class... Ts> struct Overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;

std::shared_ptr<IpuTensorDetails>
getTensorDetails(const at::TensorImpl &ipu_tensor);

// This is just a useful helper since sometimes we need to pass both keys in.
// TODO(T59880): replace XLA -> IPU
c10::DispatchKeySet dispatch_key_set{c10::DispatchKey::XLA,
                                     c10::DispatchKey::AutogradXLA};

} // namespace

poptorch_ir::Type toCompilerType(const at::ScalarType &elem_type) {
  switch (elem_type) {
  case at::ScalarType::Bool:
    return poptorch_ir::Type::BOOL;
  case at::ScalarType::Byte:
    return poptorch_ir::Type::UNSIGNED_CHAR;
  case at::ScalarType::Char:
    return poptorch_ir::Type::CHAR;
  case at::ScalarType::Float:
  case at::ScalarType::Double: // We will convert this.
    return poptorch_ir::Type::FLOAT;
  case at::ScalarType::Half:
    return poptorch_ir::Type::HALF;
  case at::ScalarType::Short:
    return poptorch_ir::Type::SHORT;
  case at::ScalarType::Int:
  case at::ScalarType::Long: // We will convert this.
    return poptorch_ir::Type::INT;
  default:
    ERROR("Unsupported tensor input type from pytorch: " << elem_type);
  }
}

poptorch_ir::Type toCompilerElementType(const at::Tensor &tensor) {
  auto dtype = tensor.dtype();
  return toCompilerType(dtype.toScalarType());
}

// Return the data size in bytes of a tensor (i.e num_elems * elem_size)
uint64_t tensorImplDataSize(const at::TensorImpl &impl) {
  auto shape = impl.sizes();
  const std::int64_t nelems = std::accumulate(shape.begin(), shape.end(), 1,
                                              std::multiplies<std::int64_t>());
  const auto elem_size = impl.itemsize();
  return nelems * elem_size;
}

// This is our own TensorImpl: this is stored in every at::Tensor of type IPU.
//
// This implementation is inspired by VulkanOpaqueTensorImpl / OpaqueTensorImpl:
// they seem to have similar needs to us.
struct IpuTensorImpl : public at::TensorImpl {
  // Shallow copy constructor (Both instances will share the same host buffer if
  // it exists). Shouldn't be called directly: use shallow_copy_and_detach()
  // instead.
  IpuTensorImpl(const IpuTensorImpl &src)
      : IpuTensorImpl(src.dtype(), src.device(),
                      src.sizes_and_strides_.sizes_arrayref(),
                      src.sizes_and_strides_.strides_arrayref(), src.details) {}

  void release_resources() override { details.reset(); }

  IpuTensorImpl(const caffe2::TypeMeta data_type, c10::Device device,
                c10::IntArrayRef sizes, c10::IntArrayRef strides,
                const std::shared_ptr<IpuTensorDetails> &details_)
      : at::TensorImpl(dispatch_key_set, data_type, device), details(details_) {
    // set_sizes must be called before stride_at because it resizes the
    // array that stores both sizes and strides.
    sizes_and_strides_.set_sizes(sizes);
    for (uint dim = 0; dim < strides.size(); ++dim) {
      sizes_and_strides_.stride_at(dim) = strides.at(dim);
    }

    set_storage_access_should_throw();
    set_has_contiguity_policy(
        at::TensorImpl::HasContiguityPolicy::CustomBehavior);
    is_non_overlapping_and_dense_ = false;
    refresh_numel();
  }

  c10::intrusive_ptr<TensorImpl>
  shallow_copy_and_detach(const c10::VariableVersion &version_counter,
                          bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<IpuTensorImpl>(*this);
    copy_tensor_metadata(
        /*src_impl=*/this,
        /*dest_impl=*/impl.get(),
        /*version_counter=*/version_counter,
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    impl->refresh_numel();

    return impl;
  }

  c10::intrusive_ptr<TensorImpl>
  shallow_copy_and_detach(c10::VariableVersion &&version_counter,
                          bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<IpuTensorImpl>(*this);
    copy_tensor_metadata(
        /*src_impl=*/this,
        /*dest_impl=*/impl.get(),
        /*version_counter=*/version_counter,
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    impl->refresh_numel();

    return impl;
  }

  bool is_contiguous_custom(c10::MemoryFormat memory_format) const override {
    UNUSED(memory_format);
    return true;
  }

  void set_size(int64_t dim, int64_t new_size) override {
    UNUSED(dim);
    UNUSED(new_size);
    AT_ERROR("IPU tensors do not have set_size");
  }

  void set_stride(int64_t dim, int64_t new_stride) override {
    UNUSED(dim);
    UNUSED(new_stride);
    AT_ERROR("IPU tensors do not have set_stride");
  }

  void set_storage_offset(int64_t storage_offset) override {
    UNUSED(storage_offset);
    AT_ERROR("IPU tensors do not have set_storage_offset");
  }

  std::shared_ptr<IpuTensorDetails> details;

private:
  const char *tensorimpl_type_name() const override { return "IpuTensorImpl"; }
};

namespace {

IpuTensorImpl *tryIpuTensorImpl(const at::Tensor &tensor) {
  return dynamic_cast<IpuTensorImpl *>(tensor.unsafeGetTensorImpl());
}

IpuTensorImpl *toIpuTensorImpl(const at::Tensor &tensor) {
  auto *ptr = tryIpuTensorImpl(tensor);
  ERROR_ON_MSG(ptr == nullptr,
               "Expected an IPU tensor but "
                   << tensor.unsafeGetTensorImpl() << " is "
                   << tensor.unsafeGetTensorImpl()->device_type());
  return ptr;
}

const IpuTensorImpl *toIpuTensorImpl(const at::TensorImpl &tensor) {
  const auto *impl = dynamic_cast<const IpuTensorImpl *>(&tensor);
  ERROR_ON_MSG(impl == nullptr, "Expected an IPU tensor but "
                                    << &tensor << " is "
                                    << tensor.device_type());
  return impl;
}

std::shared_ptr<IpuTensorDetails>
getTensorDetails(const at::TensorImpl &ipu_tensor) {
  return toIpuTensorImpl(ipu_tensor)->details;
}

// TODO(T61601) Create a proper implementation of GuardImpl
struct GuardImpl : public c10::impl::DeviceGuardImplInterface {
  // TODO(T59880): replace XLA -> IPU
  at::DeviceType type() const override { return at::DeviceType::XLA; }

  c10::Device exchangeDevice(c10::Device device) const override {
    logging::trace("exchangeDevice: current {} new {}", _current_device,
                   device);
    c10::Device old = _current_device;
    *const_cast<c10::Device *>(&_current_device) = device;
    return old;
  }

  // Called by the dispatcher every time the user passes a device type without
  // an ID to a "to()" method For example: my_tensor.to(torch.device("ipu"))
  c10::Device getDevice() const override { return _current_device; }

  void setDevice(c10::Device device) const override {
    logging::trace("setDevice: current {} new {}", _current_device, device);
    *const_cast<c10::Device *>(&_current_device) = device;
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    logging::trace("uncheckedSetDevice: current {} new {}", _current_device,
                   device);
    *const_cast<c10::Device *>(&_current_device) = device;
  }

  // Used by the autograd.
  // Streams are essentially command queues: if kernels A & B are added to the
  // same stream, A is guaranteed to have completed before B starts.
  // For A & B to be run in parallel they need to be added to different
  // streams.
  c10::Stream getStream(c10::Device device) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    UNUSED(s);
    return c10::Stream(c10::Stream::DEFAULT, getDevice());
  }

  // Used by torch::autograd::Engine::initialize_device_threads_pool
  c10::DeviceIndex deviceCount() const noexcept override { return 1; }

private:
  // TODO(T59880): replace XLA -> IPU
  c10::Device _current_device{at::DeviceType::XLA, 0};
};

// TODO(T59880): replace XLA -> IPU
C10_REGISTER_GUARD_IMPL(XLA, GuardImpl)

poptorch_ir::TensorType getTensorType(const at::ScalarType &scalar_type,
                                      std::vector<std::int64_t> sizes) {
  poptorch_ir::TensorType type;
  type.element_type = toCompilerType(scalar_type);
  type.shape = std::move(sizes);
  return type;
}
} // namespace

poptorch_ir::TensorType getTensorType(const at::Tensor &tensor) {
  return getTensorType(tensor.scalar_type(), tensor.sizes().vec());
}

uint64_t ipuTensorId(const at::Tensor &tensor) {
  return getTensorDetails(*tensor.unsafeGetTensorImpl())->tensor_id;
}

uint64_t ipuTensorId(const at::TensorImpl &tensor) {
  return toIpuTensorImpl(tensor)->details->tensor_id;
}

bool isIpuTensor(const at::Tensor &tensor) {
  return tryIpuTensorImpl(tensor) != nullptr;
}

std::string str(const at::Tensor &tensor) {
  std::stringstream ss;
  ss << "impl_ " << reinterpret_cast<void *>(tensor.unsafeGetTensorImpl());
  if (!tensor.defined()) {
    ss << " type: <undefined>";
  } else {
    auto device_type = tensor.unsafeGetTensorImpl()->device_type();
    ss << " type " << device_type;
    if (device_type == at::DeviceType::XLA) {
      auto *ipu_tensor = toIpuTensorImpl(tensor);
      ss << " ID " << ipu_tensor->details->tensor_id;
    }
    ss << " sizes " << tensor.unsafeGetTensorImpl()->sizes();
    ss << " dtype " << tensor.unsafeGetTensorImpl()->dtype();
  }
  return ss.str();
}

uint64_t tensorDataSize(const at::Tensor &tensor) {
  return tensorImplDataSize(*tensor.unsafeGetTensorImpl());
}

Buffer &getHostBuffer(const at::Tensor &ipu_tensor) {
  return getHostBuffer(*toIpuTensorImpl(ipu_tensor));
}

Buffer &getHostBuffer(const at::TensorImpl &ipu_tensor) {
  auto details = toIpuTensorImpl(ipu_tensor)->details;
  return details->getBuffer();
}

bool hasData(const at::Tensor &ipu_tensor) {
  const auto &details = *toIpuTensorImpl(*toIpuTensorImpl(ipu_tensor))->details;
  return details.hasData();
}

void errorOnZeroSizedTensor(const at::Tensor &tensor) {
  auto sizes = tensor.sizes();
  if (std::any_of(sizes.begin(), sizes.end(),
                  [](auto dim) { return dim == 0; })) {
    std::stringstream err;
    err << "Zero-sized tensors are unsupported (Got shape [";
    for (std::size_t i = 0; i < sizes.size() - 1; i++) {
      err << sizes[i] << ", ";
    }
    err << sizes[sizes.size() - 1] << "]).";
    ERROR(err.str());
  }
}

TensorStore::TensorStore() : _ipu_session(poptorch_ir::createStaticSession()) {}

std::shared_ptr<IpuTensorDetails>
TensorStore::allocateTensorDetails(c10::IntArrayRef size,
                                   at::ScalarType coerced_scalar_type,
                                   std::shared_ptr<ITensorView> view_info) {
  for (size_t dim = 0; dim < size.size(); ++dim) {
    ERROR_ON_MSG(size.at(dim) < 0, "Invalid tensor shape: dimension "
                                       << dim << " is negative ("
                                       << size.at(dim) << ")");
  }

  auto details = std::make_shared<IpuTensorDetails>(
      _next_tensor_id++, getTensorType(coerced_scalar_type, size.vec()),
      std::move(view_info));

  return details;
}

at::Tensor TensorStore::allocateTensor(c10::IntArrayRef size,
                                       c10::optional<at::ScalarType> dtype,
                                       std::shared_ptr<ITensorView> view_info,
                                       c10::optional<at::Device> device) {
  const at::ScalarType scalar_type = scalarTypeOrDefault(dtype);
  auto coerced_scalar_type = coerceToSupportedType(scalar_type);
  auto details =
      allocateTensorDetails(size, coerced_scalar_type, std::move(view_info));
  auto strides = at::detail::defaultStrides(size);

  at::Tensor output = at::detail::make_tensor<IpuTensorImpl>(
      c10::scalarTypeToTypeMeta(coerced_scalar_type),
      deviceOrDefaultIpu(device), size, strides, std::move(details));

  // TODO(T59880): replace XLA -> IPU
  ERROR_ON(output.device().type() != c10::DeviceType::XLA);

  logging::trace(
      "Created IPU tensor: id {} impl_ {} size {} strides {} dtype {}",
      ipuTensorId(output),
      reinterpret_cast<void *>(output.unsafeGetTensorImpl()), size, strides,
      coerced_scalar_type);

  if (scalar_type != coerced_scalar_type) {
    logging::warn("[DISPATCHER] Type coerced from {} to {} for tensor id {}",
                  scalar_type, coerced_scalar_type, ipuTensorId(output));
  }

  return output;
}

Buffer &TensorStore::allocateBuffer(IpuTensorDetails &details) {
  return details.getBuffer() = _ipu_session->allocate(details.type);
}

void TensorStore::allocateBuffer(const at::Tensor &ipu_tensor) {
  auto &details = *getTensorDetails(ipu_tensor);
  allocateBuffer(details);
}

void TensorStore::copyOnIpu(const at::Tensor &ipu_dest,
                            const at::Tensor &ipu_src) {
  ERROR_ON_MSG(ipu_dest.dtype() != ipu_src.dtype(),
               "Copy operations cannot cast outside of the dispatcher.");
  const auto &src_details = getTensorDetails(ipu_src);

  const auto &dest_details = getTensorDetails(ipu_dest);
  auto dest_buf = allocateBuffer(*dest_details);
  _ipu_session->copyDataOnDevice(dest_buf, src_details->getBuffer());

  ipu_dest.set_requires_grad(ipu_src.requires_grad());
}

void TensorStore::copyFromCpu(const at::Tensor &ipu_dest,
                              const at::Tensor &cpu_src) {
  logging::trace("[DISPATCHER] Copying from CPU tensor {} with data_ptr {}",
                 static_cast<void *>(cpu_src.unsafeGetTensorImpl()),
                 cpu_src.data_ptr());

  ERROR_ON(cpu_src.dtype() != ipu_dest.dtype());
  ERROR_ON(cpu_src.sizes() != ipu_dest.sizes());

  const auto &details = getTensorDetails(ipu_dest);

  auto &buff = allocateBuffer(*details);
  _ipu_session->copyDataFromCpuSource(
      buff, static_cast<const char *>(cpu_src.data_ptr()));

  ipu_dest.set_requires_grad(cpu_src.requires_grad());
}

void TensorStore::copyToCpu(const at::Tensor &cpu_dest,
                            const at::Tensor &ipu_src) {
  logging::trace("[DISPATCHER] Copying to CPU tensor {} with data_ptr {}",
                 static_cast<void *>(cpu_dest.unsafeGetTensorImpl()),
                 cpu_dest.data_ptr());

  ERROR_ON(ipu_src.dtype() != cpu_dest.dtype());
  ERROR_ON(ipu_src.sizes() != cpu_dest.sizes());

  const auto &details = getTensorDetails(ipu_src);

  _ipu_session->copyDataToCpu(static_cast<char *>(cpu_dest.data_ptr()),
                              details->getBuffer());
}

const std::shared_ptr<poptorch_ir::IIpuSession> &
TensorStore::getIpuSession() const {
  return _ipu_session;
}

void TensorStore::enableEagerMode(bool headless) {
  _ipu_session = poptorch_ir::createEagerSession(headless);
}

void TensorStore::reset() { _ipu_session = nullptr; }
std::shared_ptr<IpuTensorDetails>

getTensorDetails(const at::Tensor &ipu_tensor) {
  return getTensorDetails(*ipu_tensor.unsafeGetTensorImpl());
}

std::vector<std::shared_ptr<IpuTensorDetails>>
getTensorDetails(const std::vector<at::Tensor> &ipu_tensors) {
  std::vector<std::shared_ptr<IpuTensorDetails>> details;
  details.reserve(ipu_tensors.size());
  std::transform(
      ipu_tensors.begin(), ipu_tensors.end(), std::back_inserter(details),
      [](const auto &ipu_tensor) { return getTensorDetails(ipu_tensor); });
  return details;
}

void setTensorDetails(const at::Tensor &ipu_tensor,
                      std::shared_ptr<IpuTensorDetails> details) {
  auto *impl = dynamic_cast<IpuTensorImpl *>(ipu_tensor.unsafeGetTensorImpl());
  ERROR_ON(impl == nullptr);
  impl->set_sizes_contiguous(details->type.shape);
  impl->details = std::move(details);
}

namespace {

IpuTensorDetails::Data getBufferOrView(std::shared_ptr<ITensorView> view_info) {
  if (view_info) {
    return view_info;
  }
  return std::make_shared<Buffer>();
}

} // namespace

IpuTensorDetails::IpuTensorDetails(IpuTensorId tensor_id_,
                                   poptorch_ir::TensorType type_,
                                   std::shared_ptr<ITensorView> view_info)
    : tensor_id(tensor_id_), type(std::move(type_)),
      data(getBufferOrView(std::move(view_info))) {}

Buffer &IpuTensorDetails::getBuffer() {
  return std::visit(
      Overloaded{[](const BufferPtr &buffer) -> Buffer & { return *buffer; },
                 [](const TensorViewPtr &view) -> Buffer & {
                   UNUSED(view);
                   ERROR("Cannot get the buffer of a view tensor.");
                 }},
      data);
}
std::shared_ptr<Buffer> IpuTensorDetails::getOwningBuffer() const {
  return std::visit(
      Overloaded{[](const BufferPtr &buffer) -> BufferPtr { return buffer; },
                 [](const TensorViewPtr &view) -> BufferPtr {
                   UNUSED(view);
                   return nullptr;
                 }},
      data);
}

bool IpuTensorDetails::hasData() const {
  return std::visit(
      Overloaded{[](const BufferPtr &buffer) { return buffer->hasData(); },
                 [](const TensorViewPtr &view) { return view != nullptr; }},
      data);
}

bool IpuTensorDetails::isView() const {
  return std::holds_alternative<TensorViewPtr>(data);
}

} // namespace poptorch
