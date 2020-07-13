// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_IMPLICIT_CASTING_HPP
#define INCLUDE_POPTORCH_IMPLICIT_CASTING_HPP
#include <vector>

namespace c10 {
template <typename T> class ArrayRef;
} // namespace c10

namespace torch {
namespace jit {
template <class T> using ArrayRef = c10::ArrayRef<T>;
struct Value;
} // namespace jit
} // namespace torch

namespace poptorch {

enum class ImplicitCast { None, All, ExceptFirst, ExceptSecond, ExceptThird };

enum class ImplicitCastOutput { None, AsPromoted, AlwaysBool, AlwaysFloat };

std::vector<torch::jit::Value *>
implicitCastInputs(torch::jit::ArrayRef<torch::jit::Value *> *inputs,
                   ImplicitCast implicit_cast);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_IMPLICIT_CASTING_HPP
