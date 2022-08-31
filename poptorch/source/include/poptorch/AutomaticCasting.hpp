// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_AUTOMATIC_CASTING_HPP
#define INCLUDE_POPTORCH_AUTOMATIC_CASTING_HPP

#include <string>
#include <vector>

namespace torch {
namespace jit {
struct Graph;
} // namespace jit
} // namespace torch

namespace poptorch {

void setAutocastEnabled(bool enabled);
void setAutocastHalf(std::vector<std::string> &&ops);
void setAutocastFloat(std::vector<std::string> &&ops);
void setAutocastPromote(std::vector<std::string> &&ops);
void setAutocastDemote(std::vector<std::string> &&ops);
void automaticCasting(torch::jit::Graph *graph);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_IMPLICIT_CASTING_HPP
