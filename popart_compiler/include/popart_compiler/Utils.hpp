// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_COMPILER_UTILS_HPP
#define POPART_COMPILER_UTILS_HPP

#include <memory>
#include <string>

namespace poptorch {

// Converts a C++ string to a unique pointer of the string array; the purpose
// is to return a "string" without using the non ABI-compatible std::string
std::unique_ptr<char[]> stringToUniquePtr(const std::string &str);

} // namespace poptorch

#endif // POPART_COMPILER_UTILS_HPP
