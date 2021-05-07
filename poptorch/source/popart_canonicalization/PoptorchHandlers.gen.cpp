// DO NOT EDIT! Generated by PopTorchHandlers.py
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"
#include "PopartCanonicalizationUtils.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

namespace {

torch::jit::Node *beginipublockHandler(torch::jit::Graph *graph,
                                       torch::jit::Node *node) {
  auto x = node->input(0);
  auto t0 = constantToLong(x->node());
  auto y = node->input(1);
  auto t1 = constantToLong(y->node());
  auto z = node->input(2);
  auto t2 = constantToLong(z->node());
  // beginIpuBlock(clong(x), clong(y), clong(z))
  return createBeginIpuBlock(graph, t0, t1, t2);
}

torch::jit::Node *callcpuopHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  auto x = node->input(0);
  auto t0 = handleTensorList(x->node());
  auto s = node->input(1);
  auto t1 = constantToString(s->node());
  auto original_node = node;
  // callCpuOp(TensorList(x), cstr(s), original_node)
  return createCallCpuOp(graph, t0, t1, original_node);
}

torch::jit::Node *endforloopHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  auto output = node->input(0);
  auto inputs = node->input(1);
  auto trip_count = node->input(2);
  auto t0 = constantToLong(trip_count->node());
  // endForLoop(output, inputs, clong(trip_count))
  return createEndForLoop(graph, output, inputs, t0);
}

torch::jit::Node *identitylossHandler(torch::jit::Graph *graph,
                                      torch::jit::Node *node) {
  auto x = node->input(0);
  auto r = node->input(1);
  auto t0 = constantToInt(r->node());
  // identityloss(x, cint(r))
  return createIdentityloss(graph, {x}, t0);
}

torch::jit::Node *internalcastHandler(torch::jit::Graph *graph,
                                      torch::jit::Node *node) {
  auto tensor = node->input(0);
  auto dtype = node->input(1);
  auto t0 = constantToString(dtype->node());
  // internalCast(tensor, cstr(dtype))
  return createInternalCast(graph, tensor, t0);
}

torch::jit::Node *ipuprinttensorHandler(torch::jit::Graph *graph,
                                        torch::jit::Node *node) {
  auto x = node->input(0);
  auto s = node->input(1);
  auto t0 = constantToString(s->node());
  // printIpuTensor(x, cstr(s))
  return createPrintIpuTensor(graph, x, t0);
}

torch::jit::Node *optimizergroupHandler(torch::jit::Graph *graph,
                                        torch::jit::Node *node) {
  auto x = node->input(0);
  auto t0 = constantToLong(x->node());
  auto l = node->input(1);
  auto t1 = handleTensorList(l->node());
  // optimizerGroup(clong(x), TensorList(l))
  return createOptimizerGroup(graph, t0, t1);
}

torch::jit::Node *recomputationcheckpointHandler(torch::jit::Graph *graph,
                                                 torch::jit::Node *node) {
  auto i0 = node->input(0);
  // recomputationCheckpoint(i0)
  return createRecomputationCheckpoint(graph, i0);
}

torch::jit::Node *setavailablememoryHandler(torch::jit::Graph *graph,
                                            torch::jit::Node *node) {
  auto x = node->input(0);
  auto y = node->input(1);
  auto t0 = constantToFloat(y->node());
  // setAvailableMemory(x, cfloat(y))
  return createSetAvailableMemory(graph, x, t0);
}

torch::jit::Node *setmatmulserializationHandler(torch::jit::Graph *graph,
                                                torch::jit::Node *node) {
  auto x = node->input(0);
  auto s = node->input(1);
  auto t0 = constantToString(s->node());
  auto a = node->input(2);
  auto t1 = constantToLong(a->node());
  auto b = node->input(3);
  auto t2 = constantToInt(b->node());
  // setMatMulSerialization(x, cstr(s), clong(a), cint(b))
  return createSetMatMulSerialization(graph, x, t0, t1, t2);
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(symbols::poptorch::begin_ipu_block, beginipublockHandler);
  registerHandler(symbols::poptorch::call_cpu_op, callcpuopHandler);
  registerHandler(symbols::poptorch::end_for_loop, endforloopHandler);
  registerHandler(symbols::poptorch::identity_loss, identitylossHandler);
  registerHandler(symbols::poptorch::internal_cast, internalcastHandler);
  registerHandler(symbols::poptorch::ipu_print_tensor, ipuprinttensorHandler);
  registerHandler(symbols::poptorch::optimizer_group, optimizergroupHandler);
  registerHandler(symbols::poptorch::recomputation_checkpoint,
                  recomputationcheckpointHandler);
  registerHandler(symbols::poptorch::set_available_memory,
                  setavailablememoryHandler);
  registerHandler(symbols::poptorch::set_matmul_serialization,
                  setmatmulserializationHandler);
}

} // namespace poptorch
