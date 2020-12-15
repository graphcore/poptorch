// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace {
torch::jit::Node *log10Handler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // Log10(X) = Log(X) / Log(10)
  torch::jit::Value *x = node->input();

  // Add log(x)
  torch::jit::Node *logx = createLog(graph, {x});

  // Add log10
  const double log10_const =
      2.302585092994045684017991454684364207601101488628772976033;
  torch::jit::Node *log10 =
      createConstantFloatLike(graph, x, {log10_const}, {});

  // Add the divide.
  return createDiv(graph, {logx->output(), log10->output()});
}

torch::jit::Node *log1pHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // Log1p(x) = log(x + 1)
  torch::jit::Value *x = node->input();

  // Add the one constant
  torch::jit::Node *one = createConstantFloatLike(graph, x, {1.0}, {});

  // Add x + 1
  torch::jit::Node *add = createAdd(graph, {x, one->output()});

  // Add the log
  return createLog(graph, {add->output()});
}

torch::jit::Node *log2Handler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // Log2(X) = Log(X) / Log(2)
  torch::jit::Value *x = node->input();

  // Add log(x)
  torch::jit::Node *logx = createLog(graph, {x});

  // Add log2
  const double log2_const =
      0.693147180559945309417232121458176568075500134360255254120;
  torch::jit::Node *log2 = createConstantFloatLike(graph, x, {log2_const}, {});

  // Add the divide.
  return createDiv(graph, {logx->output(), log2->output()});
}

torch::jit::Node *logSigmoidHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  // LogSigmoid(x) = log(σ(x))

  // σ(x)
  torch::jit::Node *sig_x = createSigmoid(graph, {node->input(0)});

  // log(σ(x))
  return createLog(graph, {sig_x->output()});
}
} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::log10, log10Handler);
  registerHandler(c10::aten::log1p, log1pHandler);
  registerHandler(c10::aten::log2, log2Handler);
  registerHandler(c10::aten::log_sigmoid, logSigmoidHandler);
}

} // namespace poptorch
