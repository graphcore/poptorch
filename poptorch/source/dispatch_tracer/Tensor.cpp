// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "Tensor.hpp"

#include <ATen/ATen.h>
#include <ATen/OpaqueTensorImpl.h>

#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

namespace {

// This is just a useful helper since sometimes we need to pass both keys in.
// TODO(T59880): replace XLA -> IPU
c10::DispatchKeySet dispatch_key_set{c10::DispatchKey::XLA,
                                     c10::DispatchKey::AutogradXLA};

// Return the data size in bytes of a tensor (i.e num_elems * elem_size)
uint64_t tensorImplDataSize(const at::TensorImpl &impl) {
  auto shape = impl.sizes();
  std::int64_t nelems = std::accumulate(shape.begin(), shape.end(), 1,
                                        std::multiplies<std::int64_t>());
  auto elem_size = impl.itemsize();
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
      : IpuTensorImpl(src.dtype(), src.device(), src.tensor_id,
                      src.sizes_and_strides_.sizes_arrayref(),
                      src.sizes_and_strides_.strides_arrayref()) {
    is_parameter = src.is_parameter;
    host_buffer = src.host_buffer;
  }

  ~IpuTensorImpl() = default;

  IpuTensorImpl(const caffe2::TypeMeta data_type, c10::Device device,
                uint64_t id, c10::IntArrayRef sizes, c10::IntArrayRef strides)
      : at::TensorImpl(dispatch_key_set, data_type, device) {
    // set_sizes must be called before stride_at because it resizes the
    // array that stores both sizes and strides.
    sizes_and_strides_.set_sizes(sizes);
    for (uint dim = 0; dim < strides.size(); ++dim) {
      sizes_and_strides_.stride_at(dim) = strides.at(dim);
    }

    tensor_id = id;
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

  void copyDataFromCpuSource(const at::Tensor &cpu_src) {
    auto data_size = tensorImplDataSize(*this);
    ERROR_ON_MSG(tensorDataSize(cpu_src) != data_size,
                 "Data size mismatch between source tensor("
                     << str(cpu_src) << "= " << tensorDataSize(cpu_src)
                     << ") and destination tensor " << data_size);
    ERROR_ON_MSG(!cpu_src.is_contiguous(),
                 "Data source must be contiguous: " << str(cpu_src));
    if (!host_buffer) {
      host_buffer = std::make_shared<std::vector<char>>(data_size);
    }
    memcpy(host_buffer->data(), cpu_src.data_ptr(), data_size);
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

  bool is_parameter{false};
  // Data source for this IPU tensor. Only allocated and populated for
  // parameters and inputs.
  Buffer host_buffer;
  uint64_t tensor_id;

private:
  const char *tensorimpl_type_name() const override { return "IpuTensorImpl"; }
};

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
} // namespace

at::Tensor createIpuTensor(at::ScalarType dtype, const at::Device &device,
                           uint64_t ipu_tensor_id, c10::IntArrayRef sizes,
                           c10::IntArrayRef strides) {
  at::Tensor out = at::detail::make_tensor<IpuTensorImpl>(
      c10::scalarTypeToTypeMeta(dtype), device, ipu_tensor_id, sizes, strides);
  for (size_t dim = 0; dim < sizes.size(); ++dim) {
    ERROR_ON_MSG(sizes.at(dim) < 0, "Invalid tensor shape: dimension "
                                        << dim << " is negative ("
                                        << sizes.at(dim) << ")");
  }
  logging::trace("createIpuTensor {} impl_ {} sizes {} strides {} dtype {}",
                 out.unsafeGetTensorImpl()->device_type(),
                 reinterpret_cast<void *>(out.unsafeGetTensorImpl()), sizes,
                 strides, dtype);
  return out;
}

uint64_t ipuTensorId(const at::Tensor &tensor) {
  return toIpuTensorImpl(tensor)->tensor_id;
}

bool isIpuTensor(const at::Tensor &tensor) {
  return tryIpuTensorImpl(tensor) != nullptr;
}

void setIsParameter(at::Tensor &tensor, bool is_parameter) {
  toIpuTensorImpl(tensor)->is_parameter = is_parameter;
}

bool isParameter(const at::Tensor &tensor) {
  return toIpuTensorImpl(tensor)->is_parameter;
}

bool isParameter(const at::TensorImpl &tensor) {
  return toIpuTensorImpl(tensor)->is_parameter;
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
      ss << " ID " << ipu_tensor->tensor_id;
      if (ipu_tensor->is_parameter) {
        ss << " is_parameter";
      }
    }
    ss << " sizes " << tensor.unsafeGetTensorImpl()->sizes();
    ss << " dtype " << tensor.unsafeGetTensorImpl()->dtype();
  }
  return ss.str();
}

uint64_t tensorDataSize(const at::Tensor &tensor) {
  return tensorImplDataSize(*tensor.unsafeGetTensorImpl());
}

void copyDataFromCpuSource(at::Tensor &ipu_tensor, const at::Tensor &cpu_src) {
  toIpuTensorImpl(ipu_tensor)->copyDataFromCpuSource(cpu_src);
}

Buffer getCpuData(const at::Tensor &ipu_tensor) {
  return toIpuTensorImpl(ipu_tensor)->host_buffer;
}

Buffer getCpuData(const at::TensorImpl &ipu_tensor) {
  return toIpuTensorImpl(ipu_tensor)->host_buffer;
}

} // namespace poptorch
