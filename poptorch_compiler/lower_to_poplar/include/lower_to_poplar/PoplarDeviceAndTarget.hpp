// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_POPLAR_DEVICE_AND_TARGET_HPP_
#define POPTORCH_POPLAR_DEVICE_AND_TARGET_HPP_

#include <memory>

namespace model_runtime {
class Device;
} // namespace model_runtime

namespace poplar {
class Device;
class Target;
} // namespace poplar

namespace poptorch_ir {
class PoplarTarget {
public:
  explicit PoplarTarget(const poplar::Target &target);
  ~PoplarTarget();

  const poplar::Target &target() const;

private:
  // Access the target via model_runtime::Device to avoid cloning
  std::unique_ptr<poplar::Target> _target;
};

// NB It is safe to lose the sole instances of this while the IPU is still in
// use because, despite the prevalance of unique_ptrs, calling
// poplar::Engine::load(device) or poplar::Engine::prepare(device) yields a
// clone of the device.
class PoplarDevice {
public:
  explicit PoplarDevice(std::shared_ptr<model_runtime::Device> device);
  ~PoplarDevice();

  PoplarTarget getTarget() const;

  const poplar::Device &device() const;
  poplar::Device &device();

  // Return the default device, taking into account environment variables
  static PoplarDevice defaultDevice();

private:
  std::shared_ptr<model_runtime::Device> _device;
};

} // namespace poptorch_ir

#endif // POPTORCH_POPLAR_DEVICE_AND_TARGET_HPP_
