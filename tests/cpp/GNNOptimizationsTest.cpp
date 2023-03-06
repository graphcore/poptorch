// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE GNNOptimizationsTest
#include <boost/test/included/unit_test.hpp>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>

#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"

#define CHECK_OPS_IN_GRAPH(graph_str, op)                                      \
  BOOST_CHECK_EQUAL(occurrences(graph_str, std::string(":").append(#op)), op);

int occurrences(const std::string &graph, const std::string &phrase) {
  int occurrs = 0;
  std::string::size_type position = 0;
  while ((position = graph.find(phrase, position)) != std::string::npos) {
    occurrs++;
    position += phrase.length();
  }
  return occurrs;
}

std::string parseGraphToStr(torch::jit::Graph *graph) {
  std::stringstream output_ir_stream;
  for (auto *node : graph->nodes()) {
    node->print(output_ir_stream, 0, nullptr, true, false, false, false);
  }
  return output_ir_stream.str();
}

void checkIsReturnUpdated(torch::jit::Graph *graph) {
  torch::jit::Node *output = graph->outputs()[0]->node();
  std::stringstream output_ir_stream;
  output->print(output_ir_stream, 0, nullptr, true, false, false, false);
  // Return from scatterreduce should be replaced by squeeze from grouped
  // version.
  BOOST_CHECK_EQUAL(occurrences(output_ir_stream.str(), "squeeze"), 1);
}

BOOST_AUTO_TEST_CASE(GroupScatterReduceAndGatherNodes0) {
  auto graph = std::make_shared<torch::jit::Graph>();
  const std::string input =
      R"IR(
    graph():
        %1  : Float(requires_grad=0, device=cpu) = poptorch::tensor_constant()
        %2  : Int(requires_grad=0, device=cpu) = poptorch::tensor_constant()
        %3  : Float(requires_grad=0, device=cpu) = poptorch::tensor_constant()
        %4  : Int(requires_grad=0, device=cpu) = poptorch::tensor_constant()
        %5  : Float(2, strides=[1], requires_grad=0, device=cpu) = popart::scatterreduce[axis_size=0, axis=0, reduction=0, enable_index_broadcast=1](%1, %2)
        %6  : Float(2, strides=[1], requires_grad=0, device=cpu) = popart::scatterreduce[axis_size=0, axis=0, reduction=0, enable_index_broadcast=1](%3, %4)
        %7  : Float(2, strides=[1], requires_grad=0, device=cpu) = popart::scatterreduce[axis_size=0, axis=0, reduction=0, enable_index_broadcast=1](%5, %6)
        %8  : Float(2, strides=[1], requires_grad=0, device=cpu) = popart::gather[axis=0](%1, %2)
        %9  : Float(2, strides=[1], requires_grad=0, device=cpu) = popart::gather[axis=0](%3, %4)
        %10 : Float(2, strides=[1], requires_grad=0, device=cpu) = popart::gather[axis=0](%5, %6)
        return(%6)
  )IR";
  parseIR(input, graph.get());
  poptorch::groupScatterReduceAndGatherNodes(graph.get());
  constexpr std::size_t tensor_constant = 4;
  constexpr std::size_t unsqueeze = 8;
  constexpr std::size_t concat = 4;
  constexpr std::size_t groupedscatterreduce = 1;
  constexpr std::size_t groupedgather = 1;
  constexpr std::size_t scatterreduce = 1;
  constexpr std::size_t gather = 1;
  constexpr std::size_t slice = 4;
  constexpr std::size_t squeeze = 4;

  std::string output_ir = parseGraphToStr(graph.get());

  CHECK_OPS_IN_GRAPH(output_ir, tensor_constant);
  CHECK_OPS_IN_GRAPH(output_ir, unsqueeze);
  CHECK_OPS_IN_GRAPH(output_ir, concat);
  CHECK_OPS_IN_GRAPH(output_ir, groupedscatterreduce);
  CHECK_OPS_IN_GRAPH(output_ir, scatterreduce);
  CHECK_OPS_IN_GRAPH(output_ir, groupedgather);
  CHECK_OPS_IN_GRAPH(output_ir, gather);
  CHECK_OPS_IN_GRAPH(output_ir, slice);
  CHECK_OPS_IN_GRAPH(output_ir, squeeze);
  checkIsReturnUpdated(graph.get());
}

BOOST_AUTO_TEST_CASE(GroupScatterReduceAndGatherNodes1) {
  auto graph = std::make_shared<torch::jit::Graph>();
  const std::string input =
      R"IR(
    graph():
        %1  : Float(requires_grad=0, device=cpu) = poptorch::tensor_constant()
        %2  : Int(requires_grad=0, device=cpu) = poptorch::tensor_constant()
        %3  : Float(requires_grad=0, device=cpu) = poptorch::tensor_constant()
        %4  : Int(requires_grad=0, device=cpu) = poptorch::tensor_constant()
        %5  : Float(requires_grad=0, device=cpu) = poptorch::tensor_constant()
        %6  : Int(requires_grad=0, device=cpu) = poptorch::tensor_constant()
        %7  : Float(2, strides=[1], requires_grad=0, device=cpu) = popart::scatterreduce[axis_size=0, axis=0, reduction=0, enable_index_broadcast=1](%1, %2)
        %8  : Float(2, strides=[1], requires_grad=0, device=cpu) = popart::scatterreduce[axis_size=0, axis=0, reduction=0, enable_index_broadcast=1](%3, %4)
        %9  : Float(2, strides=[1], requires_grad=0, device=cpu) = popart::scatterreduce[axis_size=0, axis=1, reduction=0, enable_index_broadcast=1](%5, %6)
        %10 : Float(2, strides=[1], requires_grad=0, device=cpu) = popart::gather[axis=0](%1, %2)
        %11 : Float(2, strides=[1], requires_grad=0, device=cpu) = popart::gather[axis=0](%3, %4)
        %12 : Float(2, strides=[1], requires_grad=0, device=cpu) = popart::gather[axis=1](%5, %6)
        return(%8)
  )IR";
  parseIR(input, graph.get());
  poptorch::groupScatterReduceAndGatherNodes(graph.get());
  constexpr std::size_t tensor_constant = 6;
  constexpr std::size_t unsqueeze = 8;
  constexpr std::size_t concat = 4;
  constexpr std::size_t groupedscatterreduce = 1;
  constexpr std::size_t groupedgather = 1;
  constexpr std::size_t scatterreduce = 1;
  constexpr std::size_t gather = 1;
  constexpr std::size_t slice = 4;
  constexpr std::size_t squeeze = 4;

  std::string output_ir = parseGraphToStr(graph.get());

  CHECK_OPS_IN_GRAPH(output_ir, tensor_constant);
  CHECK_OPS_IN_GRAPH(output_ir, unsqueeze);
  CHECK_OPS_IN_GRAPH(output_ir, concat);
  CHECK_OPS_IN_GRAPH(output_ir, groupedscatterreduce);
  CHECK_OPS_IN_GRAPH(output_ir, scatterreduce);
  CHECK_OPS_IN_GRAPH(output_ir, groupedgather);
  CHECK_OPS_IN_GRAPH(output_ir, gather);
  CHECK_OPS_IN_GRAPH(output_ir, slice);
  CHECK_OPS_IN_GRAPH(output_ir, squeeze);
  checkIsReturnUpdated(graph.get());
}
