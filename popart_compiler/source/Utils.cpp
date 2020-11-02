// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "popart_compiler/Utils.hpp"

namespace poptorch {

std::unique_ptr<char[]> stringToUniquePtr(const std::string &str) {
  auto ptr = std::unique_ptr<char[]>(new char[str.size() + 1]);
  str.copy(ptr.get(), std::string::npos);
  ptr.get()[str.size()] = '\0';
  return ptr;
}

} // namespace poptorch
