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
struct Graph;
struct Value;
} // namespace jit
} // namespace torch

namespace poptorch {

enum class ImplicitCast {
  None,
  All,
  ExceptFirst,
  ExceptSecond,
  ExceptThird,
  ExceptFourthFifth
};

enum class ImplicitCastOutput { None, AsPromoted, AlwaysBool, AlwaysFloat };

std::vector<torch::jit::Value *>
implicitCastInputs(torch::jit::ArrayRef<torch::jit::Value *> *inputs,
                   ImplicitCast implicit_cast);

// TODO(T55228): remove after we use our own dispatch key.
// With the dispatcher we catch implicit torch casts (intercepted with
// JitDispatch::toCopyInplace) but it seems that in the case of CPU tensors,
// the returned (casted) aten tensors are not reflected in the later ops, i.e.
// we might end up with dead implicit casts in the ir which we clean with this
// pass. The actual poptorch casting is done in our canonicalization handlers
// anyway.
void removeDeadImplicitCasts(torch::jit::Graph *graph);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_IMPLICIT_CASTING_HPP
