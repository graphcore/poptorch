// DO NOT EDIT! Generated by PopTorchHandlers.py
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"
#include "PopartCanonicalizationUtils.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

namespace {

torch::jit::Node *beginAutocastHandler(torch::jit::Graph * /*graph*/,
                                       torch::jit::Node * /*node*/) {
  // <pass through>
  return nullptr;
}

torch::jit::Node *beginIpuBlockHandler(torch::jit::Graph *graph,
                                       torch::jit::Node *node) {
  auto *x = node->input(0);
  auto t0 = constantToLong(x->node());
  auto *y = node->input(1);
  auto t1 = constantToLong(y->node());
  auto *z = node->input(2);
  auto t2 = constantToLong(z->node());
  // beginIpuBlock(clong(x), clong(y), clong(z))
  return createBeginIpuBlock(graph, t0, t1, t2);
}

torch::jit::Node *beginMultiConvHandler(torch::jit::Graph * /*graph*/,
                                        torch::jit::Node * /*node*/) {
  // <pass through>
  return nullptr;
}

torch::jit::Node *callCpuOpHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  auto *x = node->input(0);
  auto t0 = handleTensorList(x->node());
  auto *s = node->input(1);
  auto t1 = constantToString(s->node());
  auto *original_node = node;
  // callCpuOp(TensorList(x), cstr(s), original_node)
  return createCallCpuOp(graph, t0, t1, original_node);
}

torch::jit::Node *endCpuOpHandler(torch::jit::Graph * /*graph*/,
                                  torch::jit::Node * /*node*/) {
  // <pass through>
  return nullptr;
}

torch::jit::Node *endForLoopHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  auto *output = node->input(0);
  auto *inputs = node->input(1);
  auto *trip_count = node->input(2);
  auto t0 = constantToLong(trip_count->node());
  // endForLoop(output, inputs, clong(trip_count))
  return createEndForLoop(graph, output, inputs, t0);
}

torch::jit::Node *endIpuBlockHandler(torch::jit::Graph * /*graph*/,
                                     torch::jit::Node * /*node*/) {
  // <pass through>
  return nullptr;
}

torch::jit::Node *endLoopBeginHandler(torch::jit::Graph * /*graph*/,
                                      torch::jit::Node * /*node*/) {
  // <pass through>
  return nullptr;
}

torch::jit::Node *identityLossHandler(torch::jit::Graph *graph,
                                      torch::jit::Node *node) {
  auto *x = node->input(0);
  auto *r = node->input(1);
  auto t0 = constantToInt(r->node());
  // identityloss(x, cint(r))
  return createIdentityloss(graph, {x}, t0);
}

torch::jit::Node *internalCastHandler(torch::jit::Graph *graph,
                                      torch::jit::Node *node) {
  auto *tensor = node->input(0);
  auto *dtype = node->input(1);
  auto t0 = constantToString(dtype->node());
  // internalCast(tensor, cstr(dtype))
  return createInternalCast(graph, tensor, t0);
}

torch::jit::Node *ipuPrintTensorHandler(torch::jit::Graph *graph,
                                        torch::jit::Node *node) {
  auto *x = node->input(0);
  auto *s = node->input(1);
  auto t0 = constantToString(s->node());
  // printIpuTensor(x, cstr(s))
  return createPrintIpuTensor(graph, x, t0);
}

torch::jit::Node *nopHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto *x = node->input(0);
  // nop(x)
  return createNop(graph, {x});
}

torch::jit::Node *optimizerGroupHandler(torch::jit::Graph *graph,
                                        torch::jit::Node *node) {
  auto *x = node->input(0);
  auto t0 = constantToLong(x->node());
  auto *l = node->input(1);
  auto t1 = handleTensorList(l->node());
  // optimizerGroup(clong(x), TensorList(l))
  return createOptimizerGroup(graph, t0, t1);
}

torch::jit::Node *popNameScopeHandler(torch::jit::Graph * /*graph*/,
                                      torch::jit::Node * /*node*/) {
  // <pass through>
  return nullptr;
}

torch::jit::Node *recomputationCheckpointHandler(torch::jit::Graph *graph,
                                                 torch::jit::Node *node) {
  auto *i0 = node->input(0);
  // recomputationCheckpoint(i0)
  return createRecomputationCheckpoint(graph, i0);
}

torch::jit::Node *restoreAutocastHandler(torch::jit::Graph * /*graph*/,
                                         torch::jit::Node * /*node*/) {
  // <pass through>
  return nullptr;
}

torch::jit::Node *setAvailableMemoryHandler(torch::jit::Graph *graph,
                                            torch::jit::Node *node) {
  auto *x = node->input(0);
  auto *y = node->input(1);
  auto t0 = constantToFloat(y->node());
  // setAvailableMemory(x, cfloat(y))
  return createSetAvailableMemory(graph, x, t0);
}

torch::jit::Node *setMatmulSerializationHandler(torch::jit::Graph *graph,
                                                torch::jit::Node *node) {
  auto *x = node->input(0);
  auto *s = node->input(1);
  auto t0 = constantToString(s->node());
  auto *a = node->input(2);
  auto t1 = constantToLong(a->node());
  auto *b = node->input(3);
  auto t2 = constantToInt(b->node());
  // setMatMulSerialization(x, cstr(s), clong(a), cint(b))
  return createSetMatMulSerialization(graph, x, t0, t1, t2 != 0);
}

torch::jit::Node *startIfTrueHandler(torch::jit::Graph * /*graph*/,
                                     torch::jit::Node * /*node*/) {
  // <pass through>
  return nullptr;
}

torch::jit::Node *suppressAutocastHandler(torch::jit::Graph * /*graph*/,
                                          torch::jit::Node * /*node*/) {
  // <pass through>
  return nullptr;
}

torch::jit::Node *updateParamInplaceHandler(torch::jit::Graph *graph,
                                            torch::jit::Node *node) {
  auto *i0 = node->input(0);
  auto *i1 = node->input(1);
  // copyvarupdate(i0, i1)
  return createCopyvarupdate(graph, {i0, i1});
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(symbols::poptorch::begin_autocast, beginAutocastHandler);
  registerHandler(symbols::poptorch::begin_ipu_block, beginIpuBlockHandler);
  registerHandler(symbols::poptorch::begin_multi_conv, beginMultiConvHandler);
  registerHandler(symbols::poptorch::call_cpu_op, callCpuOpHandler);
  registerHandler(symbols::poptorch::end_cpu_op, endCpuOpHandler);
  registerHandler(symbols::poptorch::end_for_loop, endForLoopHandler);
  registerHandler(symbols::poptorch::end_ipu_block, endIpuBlockHandler);
  registerHandler(symbols::poptorch::end_loop_begin, endLoopBeginHandler);
  registerHandler(symbols::poptorch::identity_loss, identityLossHandler);
  registerHandler(symbols::poptorch::internal_cast, internalCastHandler);
  registerHandler(symbols::poptorch::ipu_print_tensor, ipuPrintTensorHandler);
  registerHandler(symbols::poptorch::nop, nopHandler);
  registerHandler(symbols::poptorch::optimizer_group, optimizerGroupHandler);
  registerHandler(symbols::poptorch::pop_name_scope, popNameScopeHandler);
  registerHandler(symbols::poptorch::recomputation_checkpoint,
                  recomputationCheckpointHandler);
  registerHandler(symbols::poptorch::restore_autocast, restoreAutocastHandler);
  registerHandler(symbols::poptorch::set_available_memory,
                  setAvailableMemoryHandler);
  registerHandler(symbols::poptorch::set_matmul_serialization,
                  setMatmulSerializationHandler);
  registerHandler(symbols::poptorch::start_if_true, startIfTrueHandler);
  registerHandler(symbols::poptorch::suppress_autocast,
                  suppressAutocastHandler);
  registerHandler(symbols::poptorch::update_param_inplace,
                  updateParamInplaceHandler);
}

} // namespace poptorch
