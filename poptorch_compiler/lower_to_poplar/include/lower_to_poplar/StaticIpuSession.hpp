// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_LOWER_TO_POPLAR_STATIC_IPU_SESSION_HPP_
#define POPTORCH_LOWER_TO_POPLAR_STATIC_IPU_SESSION_HPP_

#include <memory>

#include "lower_to_poplar/EagerIpuSession.hpp"

namespace poptorch_ir {

class StaticIpuSession final : public IIpuSession {
public:
  Buffer allocate(const TensorType &type) override;
  void copyDataFromCpuSource(Buffer &ipu_dest, const char *cpu_data) override;
  void copyDataToCpu(char *cpu_dest, Buffer &ipu_src) override;
  void copyDataOnDevice(Buffer &dest, const Buffer &src) override;
};

} // namespace poptorch_ir

#endif // POPTORCH_LOWER_TO_POPLAR_STATIC_IPU_SESSION_HPP_
