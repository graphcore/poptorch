// DO NOT EDIT! Generated by PopAtenHandlers.py
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"
#include "PopartCanonicalizationUtils.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

namespace {

torch::jit::Node *absHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto i0 = node->input(0);
  // abs(i0)
  return createAbs(graph, {i0});
}

torch::jit::Node *acosHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto i0 = node->input(0);
  // acos(i0)
  return createAcos(graph, {i0});
}

torch::jit::Node *acoshHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto i0 = node->input(0);
  // acosh(i0)
  return createAcosh(graph, {i0});
}

torch::jit::Node *addmmHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto y = node->input(1);
  auto z = node->input(2);
  auto x = node->input(0);
  auto c1 = node->input(3);
  auto t0 = constantToFloat(c1->node());
  auto c2 = node->input(4);
  auto t1 = constantToFloat(c2->node());
  // gemm(y, z, x, cfloat(c1), cfloat(c2), 0, 0)
  return createGemm(graph, {y, z, x}, t0, t1, 0, 0);
}

torch::jit::Node *asinHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto i0 = node->input(0);
  // asin(i0)
  return createAsin(graph, {i0});
}

torch::jit::Node *asinhHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto i0 = node->input(0);
  // asinh(i0)
  return createAsinh(graph, {i0});
}

torch::jit::Node *atanHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto i0 = node->input(0);
  // atan(i0)
  return createAtan(graph, {i0});
}

torch::jit::Node *atan2Handler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto i0 = node->input(0);
  auto i1 = node->input(1);
  // atan2(i0, i1)
  return createAtan2(graph, {i0, i1});
}

torch::jit::Node *atanhHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto i0 = node->input(0);
  // atanh(i0)
  return createAtanh(graph, {i0});
}

torch::jit::Node *catHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto x = node->input(0);
  auto t0 = handleTensorList(x->node());
  auto y = node->input(1);
  auto t1 = constantToLong(y->node());
  // concat(TensorList(x), clong(y))
  return createConcat(graph, {t0}, t1);
}

torch::jit::Node *ceilHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto i0 = node->input(0);
  // ceil(i0)
  return createCeil(graph, {i0});
}

torch::jit::Node *celuHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto x = node->input(0);
  auto a = node->input(1);
  auto t0 = createDiv(graph, {x, a})->output();
  // matched expm1: sub(exp(x), 1.0)
  auto t1 = createExpm1(graph, {t0})->output();
  auto t2 = createMul(graph, {a, t1})->output();
  auto t3 = createConstantFloatLike(graph, x, {0.0}, {})->output();
  auto t4 = createMax(graph, {x, t3})->output();
  auto t5 = createMin(graph, {t3, t2})->output();
  // add(max(x, 0.0), min(0.0, mul(a, expm1(div(x, a)))))
  return createAdd(graph, {t4, t5});
}

torch::jit::Node *constantpadndHandler(torch::jit::Graph *graph,
                                       torch::jit::Node *node) {
  auto x = node->input(0);
  auto l = node->input(1);
  auto t0 = constantToLongVec(l->node());
  auto c = node->input(2);
  auto t1 = constantToFloat(c->node());
  // constantPad(x, clong_list(l), cfloat(c))
  return createConstantPad(graph, x, t0, t1);
}

torch::jit::Node *cosHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto i0 = node->input(0);
  // cos(i0)
  return createCos(graph, {i0});
}

torch::jit::Node *coshHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto i0 = node->input(0);
  // cosh(i0)
  return createCosh(graph, {i0});
}

torch::jit::Node *detachHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  auto i0 = node->input(0);
  // detach(i0)
  return createDetach(graph, {i0});
}

torch::jit::Node *divHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto i0 = node->input(0);
  auto i1 = node->input(1);
  // div(i0, i1)
  return createDiv(graph, {i0, i1});
}

torch::jit::Node *dropoutHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  auto x = node->input(0);
  auto y = node->input(1);
  auto t0 = constantToFloat(y->node());
  // dropout(x, 1, cfloat(y))
  return createDropout(graph, {x}, 1, t0);
}

torch::jit::Node *eluHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto x = node->input(0);
  auto y = node->input(1);
  auto t0 = constantToFloat(y->node());
  // elu(x, cfloat(y))
  return createElu(graph, {x}, t0);
}

torch::jit::Node *eqHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto i0 = node->input(0);
  auto i1 = node->input(1);
  // equal(i0, i1)
  return createEqual(graph, {i0, i1});
}

torch::jit::Node *erfHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto i0 = node->input(0);
  // erf(i0)
  return createErf(graph, {i0});
}

torch::jit::Node *erfcHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto x = node->input(0);
  auto t0 = createErf(graph, {x})->output();
  auto t1 = createConstantFloatLike(graph, t0, {1.0}, {})->output();
  // sub(1.0, erf(x))
  return createSub(graph, {t1, t0});
}

torch::jit::Node *expHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto i0 = node->input(0);
  // exp(i0)
  return createExp(graph, {i0});
}

torch::jit::Node *expm1Handler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto i0 = node->input(0);
  // expm1(i0)
  return createExpm1(graph, {i0});
}

torch::jit::Node *floorHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto i0 = node->input(0);
  // floor(i0)
  return createFloor(graph, {i0});
}

torch::jit::Node *fmodHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto i0 = node->input(0);
  auto i1 = node->input(1);
  // fmod(i0, i1)
  return createFmod(graph, {i0, i1});
}

torch::jit::Node *frobeniusnormHandler(torch::jit::Graph *graph,
                                       torch::jit::Node *node) {
  if (node->inputs().size() == 1) {
    auto x = node->input(0);
    auto t0 = reduceHelperDimensionCreator(x);
    // reducel2(x, dimension_list(x), 0)
    return createReducel2(graph, {x}, t0, 0);
  }
  if (node->inputs().size() == 3) {
    auto x = node->input(0);
    auto l = node->input(1);
    auto t0 = constantToLongVec(l->node());
    auto t1 = reduceHelperDimensionCreator(x, t0);
    auto c = node->input(2);
    auto t2 = constantToLong(c->node());
    // reducel2(x, dimension_list(x, clong_list(l)), clong(c))
    return createReducel2(graph, {x}, t1, t2);
  }
  ERROR("Unhandled arity for operator c10::aten::frobenius_norm");
  return nullptr;
}

torch::jit::Node *fullHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto y = node->input(1);
  auto x = node->input(0);
  auto t0 = constantToLongVec(x->node());
  auto t1 = intVectorToIrConstant(graph, t0);
  auto t2 = createExpand(graph, {y, t1})->output();
  auto t3 = node->output(0);
  auto t5 = getNodeScalarType(t3);
  // cast(expand(y, AsIr(clong_list(x))), scalar_type(output0))
  return createCast(graph, t2, t5);
}

torch::jit::Node *fulllikeHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  auto y = node->input(1);
  auto x = node->input(0);
  auto t0 = shapeFromTensor(x);
  auto t1 = intVectorToIrConstant(graph, t0);
  // expand(y, AsIr(tensor_shape(x)))
  return createExpand(graph, {y, t1});
}

torch::jit::Node *geHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto x = node->input(0);
  auto y = node->input(1);
  auto t0 = createGreater(graph, {x, y})->output();
  auto t1 = createEqual(graph, {x, y})->output();
  // logical_or(greater(x, y), equal(x, y))
  return createLogical_or(graph, {t0, t1});
}

torch::jit::Node *geluHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto i0 = node->input(0);
  // gelu(i0)
  return createGelu(graph, {i0});
}

torch::jit::Node *gtHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto i0 = node->input(0);
  auto i1 = node->input(1);
  // greater(i0, i1)
  return createGreater(graph, {i0, i1});
}

torch::jit::Node *hardshrinkHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  auto x = node->input(0);
  auto t0 = createAbs(graph, {x})->output();
  auto l = node->input(1);
  auto t1 = createAbs(graph, {l})->output();
  auto t2 = createGreater(graph, {t0, t1})->output();
  auto t3 = createConstantFloatLike(graph, x, {0.0}, {})->output();
  // where(greater(abs(x), abs(l)), x, 0.0)
  return createWhere(graph, {t2, x, t3});
}

torch::jit::Node *hardtanhHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  auto x = node->input(0);
  auto b = node->input(2);
  auto t0 = constantToFloat(b->node());
  auto a = node->input(1);
  auto t1 = constantToFloat(a->node());
  // clip(x, cfloat(b), cfloat(a))
  return createClip(graph, {x}, t0, t1);
}

torch::jit::Node *hingeembeddinglossHandler(torch::jit::Graph *graph,
                                            torch::jit::Node *node) {
  auto y = node->input(1);
  auto t0 = createConstantFloatLike(graph, y, {-1.0}, {})->output();
  auto t1 = createEqual(graph, {y, t0})->output();
  auto delta = node->input(2);
  auto x = node->input(0);
  auto t2 = createSub(graph, {delta, x})->output();
  auto t3 = createConstantFloatLike(graph, t2, {0.0}, {})->output();
  auto t4 = createMax(graph, {t3, t2})->output();
  auto t5 = createConstantFloatLike(graph, y, {1.0}, {})->output();
  auto t6 = createEqual(graph, {y, t5})->output();
  auto t7 = createWhere(graph, {t6, x, t3})->output();
  auto t8 = createWhere(graph, {t1, t4, t7})->output();
  auto red = node->input(3);
  auto t9 = constantToLong(red->node());
  auto t10 = convertReduceToPopart(t9);
  // identityloss(where(equal(y, -1.0), max(0.0, sub(delta, x)),
  // where(equal(y, 1.0), x, 0.0)), reduction(clong(red)))
  return createIdentityloss(graph, {t8}, t10);
}

torch::jit::Node *isnanHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto i0 = node->input(0);
  // isnan(i0)
  return createIsnan(graph, {i0});
}

torch::jit::Node *l1lossHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  auto x = node->input(0);
  auto y = node->input(1);
  auto t0 = createSub(graph, {x, y})->output();
  auto red = node->input(2);
  auto t1 = constantToLong(red->node());
  auto t2 = convertReduceToPopart(t1);
  auto t3 = createL1loss(graph, {t0}, 1.0, t2)->output();
  // identityloss(l1loss(sub(x, y), 1.0, reduction(clong(red))), 2)
  return createIdentityloss(graph, {t3}, 2);
}

torch::jit::Node *leHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto x = node->input(0);
  auto y = node->input(1);
  auto t0 = createLess(graph, {x, y})->output();
  auto t1 = createEqual(graph, {x, y})->output();
  // logical_or(less(x, y), equal(x, y))
  return createLogical_or(graph, {t0, t1});
}

torch::jit::Node *leakyreluHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  auto x = node->input(0);
  auto y = node->input(1);
  auto t0 = constantToFloat(y->node());
  // leakyrelu(x, cfloat(y))
  return createLeakyrelu(graph, {x}, t0);
}

torch::jit::Node *logHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto i0 = node->input(0);
  // log(i0)
  return createLog(graph, {i0});
}

torch::jit::Node *log10Handler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto x = node->input(0);
  auto t0 = createLog(graph, {x})->output();
  auto t1 =
      createConstantFloatLike(graph, t0, {2.302585092994046}, {})->output();
  // div(log(x), 2.302585092994046)
  return createDiv(graph, {t0, t1});
}

torch::jit::Node *log1pHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto i0 = node->input(0);
  // log1p(i0)
  return createLog1p(graph, {i0});
}

torch::jit::Node *log2Handler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto x = node->input(0);
  auto t0 = createLog(graph, {x})->output();
  auto t1 =
      createConstantFloatLike(graph, t0, {0.6931471805599453}, {})->output();
  // div(log(x), 0.6931471805599453)
  return createDiv(graph, {t0, t1});
}

torch::jit::Node *logsigmoidHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  auto x = node->input(0);
  auto t0 = createSigmoid(graph, {x})->output();
  // log(sigmoid(x))
  return createLog(graph, {t0});
}

torch::jit::Node *logicalnotHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  auto i0 = node->input(0);
  // logical_not(i0)
  return createLogical_not(graph, {i0});
}

torch::jit::Node *ltHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto i0 = node->input(0);
  auto i1 = node->input(1);
  // less(i0, i1)
  return createLess(graph, {i0, i1});
}

torch::jit::Node *marginrankinglossHandler(torch::jit::Graph *graph,
                                           torch::jit::Node *node) {
  auto y = node->input(2);
  auto t0 = createNeg(graph, {y})->output();
  auto x1 = node->input(0);
  auto x2 = node->input(1);
  auto t1 = createSub(graph, {x1, x2})->output();
  auto t2 = createMul(graph, {t0, t1})->output();
  auto margin = node->input(3);
  auto t3 = createAdd(graph, {t2, margin})->output();
  auto t4 = createConstantFloatLike(graph, t3, {0.0}, {})->output();
  auto t5 = createMax(graph, {t3, t4})->output();
  auto red = node->input(4);
  auto t6 = constantToLong(red->node());
  auto t7 = convertReduceToPopart(t6);
  // identityloss(max(add(mul(neg(y), sub(x1, x2)), margin), 0.0),
  // reduction(clong(red)))
  return createIdentityloss(graph, {t5}, t7);
}

torch::jit::Node *maskedfillHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  auto i1 = node->input(1);
  auto i2 = node->input(2);
  auto i0 = node->input(0);
  // where(i1, i2, i0)
  return createWhere(graph, {i1, i2, i0});
}

torch::jit::Node *maxHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  if (node->inputs().size() == 1) {
    auto x = node->input(0);
    auto t0 = reduceHelperDimensionCreator(x);
    // reducemax(x, dimension_list(x), 0)
    return createReducemax(graph, {x}, t0, 0);
  }
  if (node->inputs().size() == 2) {
    auto i0 = node->input(0);
    auto i1 = node->input(1);
    // max(i0, i1)
    return createMax(graph, {i0, i1});
  }
  ERROR("Unhandled arity for operator c10::aten::max");
  return nullptr;
}

torch::jit::Node *minHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  if (node->inputs().size() == 1) {
    auto x = node->input(0);
    auto t0 = reduceHelperDimensionCreator(x);
    // reducemin(x, dimension_list(x), 0)
    return createReducemin(graph, {x}, t0, 0);
  }
  if (node->inputs().size() == 2) {
    auto i0 = node->input(0);
    auto i1 = node->input(1);
    // min(i0, i1)
    return createMin(graph, {i0, i1});
  }
  ERROR("Unhandled arity for operator c10::aten::min");
  return nullptr;
}

torch::jit::Node *mselossHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  auto x = node->input(0);
  auto y = node->input(1);
  auto t0 = createSub(graph, {x, y})->output();
  auto t1 = createMul(graph, {t0, t0})->output();
  auto red = node->input(2);
  auto t2 = constantToLong(red->node());
  auto t3 = convertReduceToPopart(t2);
  // identityloss(mul(sub(x, y), sub(x, y)), reduction(clong(red)))
  return createIdentityloss(graph, {t1}, t3);
}

torch::jit::Node *neHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto x = node->input(0);
  auto y = node->input(1);
  auto t0 = createEqual(graph, {x, y})->output();
  // logical_not(equal(x, y))
  return createLogical_not(graph, {t0});
}

torch::jit::Node *negHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto i0 = node->input(0);
  // neg(i0)
  return createNeg(graph, {i0});
}

torch::jit::Node *nlllossHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  auto x = node->input(0);
  auto y = node->input(1);
  auto red = node->input(3);
  auto t0 = constantToLong(red->node());
  auto t1 = convertReduceToPopart(t0);
  auto ignore_index = node->input(4);
  auto t2 = constantToInt(ignore_index->node());
  auto t3 = createNllloss(graph, {x, y}, t1, t2, true)->output();
  // identityloss(nllloss(x, y, reduction(clong(red)), cint(ignore_index), 1),
  // 2)
  return createIdentityloss(graph, {t3}, 2);
}

torch::jit::Node *normalInPlaceHandler(torch::jit::Graph *graph,
                                       torch::jit::Node *node) {
  auto x = node->input(0);
  auto t0 = shapeFromTensor(x);
  auto c1 = node->input(1);
  auto t1 = constantToFloat(c1->node());
  auto c2 = node->input(2);
  auto t2 = constantToFloat(c2->node());
  // randomNormal(x, tensor_shape(x), cfloat(c1), cfloat(c2))
  return createRandomNormal(graph, {x}, t0, t1, t2);
}

torch::jit::Node *pixelshuffleHandler(torch::jit::Graph *graph,
                                      torch::jit::Node *node) {
  auto x = node->input(0);
  auto y = node->input(1);
  auto t0 = constantToLong(y->node());
  // depthtospace(x, clong(y), "CRD")
  return createDepthtospace(graph, {x}, t0, "CRD");
}

torch::jit::Node *powHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto i0 = node->input(0);
  auto i1 = node->input(1);
  // pow(i0, i1)
  return createPow(graph, {i0, i1});
}

torch::jit::Node *preluHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto i0 = node->input(0);
  auto i1 = node->input(1);
  // prelu(i0, i1)
  return createPrelu(graph, {i0, i1});
}

torch::jit::Node *randHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto x = node->input(0);
  auto t0 = node->output(0);
  auto t2 = shapeFromTensor(t0);
  auto t3 = getNodeScalarType(t0);
  // randomUniform(x, tensor_shape(output0), 1.0, 0.0, scalar_type(output0))
  return createRandomUniform(graph, x, t2, 1.0, 0.0, t3);
}

torch::jit::Node *randnHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto t0 = node->output(0);
  auto t2 = shapeFromTensor(t0);
  auto t3 = getNodeScalarType(t0);
  // randomNormal({}, tensor_shape(output0), 0.0, 1.0, scalar_type(output0))
  return createRandomNormal(graph, {}, t2, 0.0, 1.0, t3);
}

torch::jit::Node *reciprocalHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  auto i0 = node->input(0);
  // reciprocal(i0)
  return createReciprocal(graph, {i0});
}

torch::jit::Node *reflectionpad1dHandler(torch::jit::Graph *graph,
                                         torch::jit::Node *node) {
  auto x = node->input(0);
  auto y = node->input(1);
  auto t0 = constantToLongVec(y->node());
  // reflectionPad(x, clong_list(y))
  return createReflectionPad(graph, x, t0);
}

torch::jit::Node *reluHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto i0 = node->input(0);
  // relu(i0)
  return createRelu(graph, {i0});
}

torch::jit::Node *remainderHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  auto i0 = node->input(0);
  auto i1 = node->input(1);
  // remainder(i0, i1)
  return createRemainder(graph, {i0, i1});
}

torch::jit::Node *replicationpad1dHandler(torch::jit::Graph *graph,
                                          torch::jit::Node *node) {
  auto x = node->input(0);
  auto y = node->input(1);
  auto t0 = constantToLongVec(y->node());
  // edgePad(x, clong_list(y))
  return createEdgePad(graph, x, t0);
}

torch::jit::Node *roundHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto i0 = node->input(0);
  // round(i0)
  return createRound(graph, {i0});
}

torch::jit::Node *rsqrtHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto x = node->input(0);
  auto t0 = createSqrt(graph, {x})->output();
  // matched reciprocal: div(1.0, x)
  // reciprocal(sqrt(x))
  return createReciprocal(graph, {t0});
}

torch::jit::Node *rsubHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto y = node->input(1);
  auto x = node->input(0);
  // sub(y, x)
  return createSub(graph, {y, x});
}

torch::jit::Node *seluHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto x = node->input(0);
  // selu(x, 1.6732632423543772, 1.0507009873554805)
  return createSelu(graph, {x}, 1.6732632423543772, 1.0507009873554805);
}

torch::jit::Node *sigmoidHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  auto i0 = node->input(0);
  // sigmoid(i0)
  return createSigmoid(graph, {i0});
}

torch::jit::Node *signHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto i0 = node->input(0);
  // sign(i0)
  return createSign(graph, {i0});
}

torch::jit::Node *siluHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto x = node->input(0);
  auto t0 = createSigmoid(graph, {x})->output();
  // mul(x, sigmoid(x))
  return createMul(graph, {x, t0});
}

torch::jit::Node *sinHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto i0 = node->input(0);
  // sin(i0)
  return createSin(graph, {i0});
}

torch::jit::Node *sinhHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto i0 = node->input(0);
  // sinh(i0)
  return createSinh(graph, {i0});
}

torch::jit::Node *smoothl1lossHandler(torch::jit::Graph *graph,
                                      torch::jit::Node *node) {
  auto beta = node->input(3);
  auto x = node->input(0);
  auto y = node->input(1);
  auto t0 = createSub(graph, {x, y})->output();
  auto t1 = createAbs(graph, {t0})->output();
  auto t2 = createGreater(graph, {beta, t1})->output();
  auto t3 = createConstantFloatLike(graph, t1, {0.5}, {})->output();
  auto t4 = createMul(graph, {t3, t1})->output();
  auto t5 = createMul(graph, {t4, t1})->output();
  auto t6 = createDiv(graph, {t5, beta})->output();
  auto t7 = createMul(graph, {t3, beta})->output();
  auto t8 = createSub(graph, {t1, t7})->output();
  auto t9 = createWhere(graph, {t2, t6, t8})->output();
  auto red = node->input(2);
  auto t10 = constantToLong(red->node());
  auto t11 = convertReduceToPopart(t10);
  // identityloss(where(greater(beta, abs(sub(x, y))), div(mul(mul(0.5,
  // abs(sub(x, y))), abs(sub(x, y))), beta), sub(abs(sub(x, y)), mul(0.5,
  // beta))), reduction(clong(red)))
  return createIdentityloss(graph, {t9}, t11);
}

torch::jit::Node *softmarginlossHandler(torch::jit::Graph *graph,
                                        torch::jit::Node *node) {
  auto y = node->input(1);
  auto t0 = createNeg(graph, {y})->output();
  auto x = node->input(0);
  auto t1 = createMul(graph, {t0, x})->output();
  auto t2 = createExp(graph, {t1})->output();
  // matched log1p: log(add(1.0, x))
  auto t3 = createLog1p(graph, {t2})->output();
  auto red = node->input(2);
  auto t4 = constantToLong(red->node());
  auto t5 = convertReduceToPopart(t4);
  // identityloss(log1p(exp(mul(neg(y), x))), reduction(clong(red)))
  return createIdentityloss(graph, {t3}, t5);
}

torch::jit::Node *softplusHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  auto threshold = node->input(2);
  auto x = node->input(0);
  auto b = node->input(1);
  auto t0 = createMul(graph, {x, b})->output();
  auto t1 = createLess(graph, {threshold, t0})->output();
  auto t2 = createMul(graph, {b, x})->output();
  auto t3 = createExp(graph, {t2})->output();
  // matched log1p: log(add(1.0, x))
  auto t4 = createLog1p(graph, {t3})->output();
  // matched div: mul(div(1.0, y), x)
  auto t5 = createDiv(graph, {t4, b})->output();
  // where(less(threshold, mul(x, b)), x, div(log1p(exp(mul(b, x))), b))
  return createWhere(graph, {t1, x, t5});
}

torch::jit::Node *softshrinkHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  auto x = node->input(0);
  auto l = node->input(1);
  auto t0 = createNeg(graph, {l})->output();
  auto t1 = createLess(graph, {x, t0})->output();
  auto t2 = createAdd(graph, {x, l})->output();
  auto t3 = createGreater(graph, {x, l})->output();
  auto t4 = createSub(graph, {x, l})->output();
  auto t5 = createConstantFloatLike(graph, t4, {0.0}, {})->output();
  auto t6 = createWhere(graph, {t3, t4, t5})->output();
  // where(less(x, neg(l)), add(x, l), where(greater(x, l), sub(x, l), 0.0))
  return createWhere(graph, {t1, t2, t6});
}

torch::jit::Node *sqrtHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto i0 = node->input(0);
  // sqrt(i0)
  return createSqrt(graph, {i0});
}

torch::jit::Node *squareHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  auto x = node->input(0);
  // mul(x, x)
  return createMul(graph, {x, x});
}

torch::jit::Node *subHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto x = node->input(0);
  auto y = node->input(1);
  auto a = node->input(2);
  auto t0 = createMul(graph, {y, a})->output();
  auto t1 = hasUnityValue(a) ? y : t0;
  // sub(x, alpha(y, a, mul(y, a)))
  return createSub(graph, {x, t1});
}

torch::jit::Node *tHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto i0 = node->input(0);
  // transpose(i0, {})
  return createTranspose(graph, {i0}, {});
}

torch::jit::Node *tanHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto i0 = node->input(0);
  // tan(i0)
  return createTan(graph, {i0});
}

torch::jit::Node *tanhHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto i0 = node->input(0);
  // tanh(i0)
  return createTanh(graph, {i0});
}

torch::jit::Node *thresholdHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  auto x = node->input(0);
  auto threshold = node->input(1);
  auto t0 = createGreater(graph, {x, threshold})->output();
  auto val = node->input(2);
  // where(greater(x, threshold), x, val)
  return createWhere(graph, {t0, x, val});
}

torch::jit::Node *topkHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto x = node->input(0);
  auto c = node->input(1);
  auto t0 = c->node();
  t0->t_(c10::attr::value, t0->t(c10::attr::value).to(at::ScalarType::Long));
  t0->output()->inferTypeFrom(t0->t(c10::attr::value));
  auto t1 = t0->output();
  auto l = node->input(2);
  auto t2 = x->type()->expect<c10::TensorType>();
  auto t3 = handleDimensionParam(l, t2);
  // topk(x, inplace_cast<long>(c), dimension(l, TensorType(x)))
  return createTopk(graph, {x, t1}, t3);
}

torch::jit::Node *uniformInPlaceHandler(torch::jit::Graph *graph,
                                        torch::jit::Node *node) {
  auto x = node->input(0);
  auto t0 = shapeFromTensor(x);
  auto b = node->input(2);
  auto t1 = constantToFloat(b->node());
  auto a = node->input(1);
  auto t2 = constantToFloat(a->node());
  // randomUniform(x, tensor_shape(x), cfloat(b), cfloat(a))
  return createRandomUniform(graph, x, t0, t1, t2);
}

torch::jit::Node *whereHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto i0 = node->input(0);
  auto i1 = node->input(1);
  auto i2 = node->input(2);
  // where(i0, i1, i2)
  return createWhere(graph, {i0, i1, i2});
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::abs, absHandler);
  registerHandler(c10::aten::acos, acosHandler);
  registerHandler(c10::aten::acosh, acoshHandler);
  registerHandler(c10::aten::addmm, addmmHandler);
  registerHandler(c10::aten::asin, asinHandler);
  registerHandler(c10::aten::asinh, asinhHandler);
  registerHandler(c10::aten::atan, atanHandler);
  registerHandler(c10::aten::atan2, atan2Handler);
  registerHandler(c10::aten::atanh, atanhHandler);
  registerHandler(c10::aten::cat, catHandler);
  registerHandler(c10::aten::ceil, ceilHandler);
  registerHandler(c10::aten::celu, celuHandler);
  registerHandler(c10::aten::constant_pad_nd, constantpadndHandler);
  registerHandler(c10::aten::cos, cosHandler);
  registerHandler(c10::aten::cosh, coshHandler);
  registerHandler(c10::aten::detach, detachHandler);
  registerHandler(c10::aten::div, divHandler);
  registerHandler(c10::aten::dropout, dropoutHandler);
  registerHandler(c10::aten::dropout_, dropoutHandler);
  registerHandler(c10::aten::elu, eluHandler);
  registerHandler(c10::aten::elu_, eluHandler);
  registerHandler(c10::aten::eq, eqHandler);
  registerHandler(c10::aten::erf, erfHandler);
  registerHandler(c10::aten::erfc, erfcHandler);
  registerHandler(c10::aten::exp, expHandler);
  registerHandler(c10::aten::expm1, expm1Handler);
  registerHandler(c10::aten::floor, floorHandler);
  registerHandler(c10::aten::fmod, fmodHandler);
  registerHandler(c10::aten::frobenius_norm, frobeniusnormHandler);
  registerHandler(c10::aten::full, fullHandler);
  registerHandler(c10::aten::full_like, fulllikeHandler);
  registerHandler(c10::aten::ge, geHandler);
  registerHandler(c10::aten::gelu, geluHandler);
  registerHandler(c10::aten::gt, gtHandler);
  registerHandler(c10::aten::hardshrink, hardshrinkHandler);
  registerHandler(c10::aten::hardtanh, hardtanhHandler);
  registerHandler(c10::aten::hardtanh_, hardtanhHandler);
  registerHandler(c10::aten::hinge_embedding_loss, hingeembeddinglossHandler);
  registerHandler(c10::aten::isnan, isnanHandler);
  registerHandler(c10::aten::l1_loss, l1lossHandler);
  registerHandler(c10::aten::le, leHandler);
  registerHandler(c10::aten::leaky_relu, leakyreluHandler);
  registerHandler(c10::aten::leaky_relu_, leakyreluHandler);
  registerHandler(c10::aten::log, logHandler);
  registerHandler(c10::aten::log10, log10Handler);
  registerHandler(c10::aten::log1p, log1pHandler);
  registerHandler(c10::aten::log2, log2Handler);
  registerHandler(c10::aten::log_sigmoid, logsigmoidHandler);
  registerHandler(c10::aten::logical_not, logicalnotHandler);
  registerHandler(c10::aten::lt, ltHandler);
  registerHandler(c10::aten::margin_ranking_loss, marginrankinglossHandler);
  registerHandler(c10::aten::masked_fill, maskedfillHandler);
  registerHandler(c10::aten::masked_fill_, maskedfillHandler);
  registerHandler(c10::aten::max, maxHandler);
  registerHandler(c10::aten::min, minHandler);
  registerHandler(c10::aten::mse_loss, mselossHandler);
  registerHandler(c10::aten::ne, neHandler);
  registerHandler(c10::aten::neg, negHandler);
  registerHandler(c10::aten::nll_loss, nlllossHandler);
  registerHandler(c10::aten::normal_, normalInPlaceHandler);
  registerHandler(c10::aten::pixel_shuffle, pixelshuffleHandler);
  registerHandler(c10::aten::pow, powHandler);
  registerHandler(c10::aten::prelu, preluHandler);
  registerHandler(c10::aten::prelu_, preluHandler);
  registerHandler(c10::aten::rand, randHandler);
  registerHandler(c10::aten::randn, randnHandler);
  registerHandler(c10::aten::reciprocal, reciprocalHandler);
  registerHandler(c10::aten::reflection_pad1d, reflectionpad1dHandler);
  registerHandler(c10::aten::reflection_pad2d, reflectionpad1dHandler);
  registerHandler(c10::aten::relu, reluHandler);
  registerHandler(c10::aten::relu_, reluHandler);
  registerHandler(c10::aten::remainder, remainderHandler);
  registerHandler(c10::aten::replication_pad1d, replicationpad1dHandler);
  registerHandler(c10::aten::replication_pad2d, replicationpad1dHandler);
  registerHandler(c10::aten::replication_pad3d, replicationpad1dHandler);
  registerHandler(c10::aten::round, roundHandler);
  registerHandler(c10::aten::rsqrt, rsqrtHandler);
  registerHandler(c10::aten::rsub, rsubHandler);
  registerHandler(c10::aten::selu, seluHandler);
  registerHandler(c10::aten::selu_, seluHandler);
  registerHandler(c10::aten::sigmoid, sigmoidHandler);
  registerHandler(c10::aten::sign, signHandler);
  registerHandler(c10::aten::silu, siluHandler);
  registerHandler(c10::aten::sin, sinHandler);
  registerHandler(c10::aten::sinh, sinhHandler);
  registerHandler(c10::aten::smooth_l1_loss, smoothl1lossHandler);
  registerHandler(c10::aten::soft_margin_loss, softmarginlossHandler);
  registerHandler(c10::aten::softplus, softplusHandler);
  registerHandler(c10::aten::softshrink, softshrinkHandler);
  registerHandler(c10::aten::sqrt, sqrtHandler);
  registerHandler(c10::aten::square, squareHandler);
  registerHandler(c10::aten::sub, subHandler);
  registerHandler(c10::aten::t, tHandler);
  registerHandler(c10::aten::tan, tanHandler);
  registerHandler(c10::aten::tanh, tanhHandler);
  registerHandler(c10::aten::threshold, thresholdHandler);
  registerHandler(c10::aten::topk, topkHandler);
  registerHandler(c10::aten::uniform_, uniformInPlaceHandler);
  registerHandler(c10::aten::where, whereHandler);
  registerHandler(c10::aten::where_, whereHandler);
}

} // namespace poptorch
