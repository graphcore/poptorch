// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <limits>

#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

#include "../PoptorchSymbols.hpp"
#include "poptorch/Utils.hpp"

#include <ATen/ATen.h>

namespace poptorch {
namespace {

torch::jit::Node *reduceHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // Reductions have three overloads. The first is:
  // aten::mean(Tensor self, int[] dim, int keepdim, Tensor? out)) -> tensor

  // The second is:
  // aten::mean(Tensor self, int? dtype)) -> tensor

  // The third is for boolean reductions
  // aten::all(Tensor self) -> tensor

  torch::jit::Symbol const kind = node->kind();
  torch::jit::Value *input = node->input(0);

  // sum and prod works even for bool types in PyTorch
  auto tensor_type = input->type()->expect<c10::TensorType>();
  if (tensor_type->scalarType() == at::ScalarType::Bool) {
    auto *cast_node = createCast(graph, input, c10::ScalarType::Int);
    input = cast_node->output();
  }

  std::vector<std::int64_t> axes{};
  std::int64_t keepdim = 0;

  // Case 2/3 or case 1 with no dimension specified.
  const size_t case_2_3 =
      (kind == c10::aten::any || kind == c10::aten::all) ? 1 : 2;
  bool flatten = node->inputs().size() == case_2_3;
  if (!flatten) {
    // Case 1.
    // Sometimes the dimensions are just one int.

    if (node->input(1)->node()->kind() == symbols::poptorch::tensor_constant) {
      axes.push_back(constantToLong(node->input(1)->node()));
    } else {
      axes = constantToLongVec(node->input(1)->node());
      // No dimension specified: this is actually a case 1.
      if (axes.empty()) {
        flatten = true;
      }
    }
    keepdim = constantToLong(node->input(2)->node());
  }
  if (flatten) {
    // Need to use reshape as "Flatten" is for 2D output
    auto numels_optional = tensor_type->numel();
    ERROR_ON(!numels_optional);
    input =
        createReshape(graph, input, {static_cast<int64_t>(*numels_optional)})
            ->output();
    axes = {0};
    keepdim = 0;
  }

  // Output the correct reduction.
  if (kind == c10::aten::prod) {
    return createReduceprod(graph, {input}, axes, keepdim);
  }
  if (kind == c10::aten::mean) {
    return createReducemean(graph, {input}, axes, keepdim);
  }
  if (kind == c10::aten::sum) {
    return createReducesum(graph, {input}, axes, keepdim);
  }
  if (kind == c10::aten::logsumexp) {
    return createReducelogsumexp(graph, {input}, axes, keepdim);
  }
  if (kind == c10::aten::all) {
    auto *t0 = createAbs(graph, {input})->output();
    auto *t1 = createReducemin(graph, {t0}, axes, keepdim)->output();
    return createCast(graph, t1, at::ScalarType::Bool);
  }
  if (kind == c10::aten::any) {
    auto *t0 = createAbs(graph, {input})->output();
    auto *t1 = createReducemax(graph, {t0}, axes, keepdim)->output();
    return createCast(graph, t1, at::ScalarType::Bool);
  }
  ERROR("Popart Canonicalisation: UNREACHABLE reached in reductions.");
}

torch::jit::Node *reduceMedianHandler(torch::jit::Graph *graph,
                                      torch::jit::Node *node) {
  auto *input = node->input(0);
  std::vector<std::int64_t> axes;
  std::int64_t keepdim = 0;

  torch::jit::Node *output;

  if (node->inputs().size() == 1) {
    // aten::median(Tensor self) -> Tensor
    axes = reduceHelperDimensionCreator(input);
    auto *reduced = createReducemedian(graph, {input}, axes, keepdim);
    reduced->eraseOutput(1);
    output = reduced;
  } else {
    // aten::median(Tensor self, int dim, bool keepdim)
    //             -> (Tensor values, Tensor indices)
    axes.push_back(constantToLong(node->input(1)->node()));
    keepdim = constantToLong(node->input(2)->node());
    output = createReducemedian(graph, {input}, axes, keepdim);
  }

  return output;
}

torch::jit::Node *aMinMaxHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  // aten::max(Tensor self, int[] dim, int keepdim)
  // aten::min(Tensor self, int[] dim, int keepdim)
  auto *input = node->input(0);
  auto axes = constantToLongVec(node->input(1)->node());
  auto keepdim = constantToLong(node->input(2)->node());

  if (axes.empty()) {
    input = createFlatten(graph, {input}, 0)->output();
    axes = {1};
  }

  if (node->kind() == c10::aten::amax) {
    return createReducemax(graph, {input}, axes, keepdim);
  }
  return createReducemin(graph, {input}, axes, keepdim);
}

torch::jit::Node *argMinMaxHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  //  aten::argmin(Tensor in, int? dim, int keep_dims) -> Tensor
  //  aten::argmax(Tensor in, int? dim, int keep_dims) -> Tensor
  // dim (int) - the dimension to reduce. If None, the argmax
  //             of the flattened input is returned.

  torch::jit::Symbol const kind = node->kind();
  torch::jit::Value *input = node->input(0);

  std::optional<std::int64_t> dim;
  if (node->input(1)->node()->kind() == symbols::poptorch::tensor_constant) {
    dim = constantToLong(node->input(1)->node());
  }

  std::int64_t const keep_dim = constantToLong(node->input(2)->node());

  // If dim is not provided we will flatten input so just use 0 in that
  // case.
  std::int64_t dim_to_use = 1;

  // Check if dim is NONE.
  if (!dim) {
    torch::jit::Node *flatten = createFlatten(graph, {node->input(0)}, 0);
    input = flatten->output();
  } else {
    dim_to_use = *dim;
  }

  torch::jit::Node *indices;
  // Create the actual argmax/argmin.
  if (kind == c10::aten::argmax) {
    indices = createArgmax(graph, {input}, dim_to_use, keep_dim);
  } else {
    indices = createArgmin(graph, {input}, dim_to_use, keep_dim);
  }
  // Note: these ops return int64, so we need to cast them to int
  return createCast(graph, indices->output(), c10::ScalarType::Int);
}

torch::jit::Node *argsortHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  auto *x = node->input(0);
  auto t0 = x->type()->expect<c10::TensorType>();
  std::vector<std::int64_t> shape = shapeFromTensor(node->input(0));
  auto dim = handleDimensionParam(node->input(1), t0);

  auto *size = createConstantLong(graph, {shape[dim]}, {1})->output();

  auto *topk = createTopk(graph, {x, size}, dim);
  auto *indices = topk->output(1);

  // Onnx will output the indices long, so use a cast to revert the type.
  // PopART will remove it as an identity when topk resolves to output an int.
  indices = createCast(graph, indices, c10::ScalarType::Int)->output();

  auto descending = constantToBool(node->input(2)->node());
  if (descending) {
    return indices->node();
  }

  const std::vector<int64_t> dims{dim};
  return createReverse(graph, {indices}, dims);
}

torch::jit::Node *minMaxWithIndicesHandler(torch::jit::Graph *graph,
                                           torch::jit::Node *node) {
  auto *x = node->input(0);
  auto t0 = x->type()->expect<c10::TensorType>();
  const std::vector<std::int64_t> shape = shapeFromTensor(x);
  torch::jit::Value *values;
  torch::jit::Value *indices;
  if (shape.empty()) {
    values = createIdentity(graph, {x})->output();
    indices = createConstantInt(graph, {0}, {})->output();
  } else {
    auto dim = handleDimensionParam(node->input(1), t0);
    auto keepdim = constantToBool(node->input(2)->node());
    const bool negate = node->kind() == c10::aten::min;

    if (negate) {
      x = createNeg(graph, {x})->output();
    }

    auto *one = tensorToConstant(graph, at::tensor(1L))->output();
    auto *result = createTopk(graph, {x, one}, dim);
    values = result->output(0);
    indices = result->output(1);
    // TopK returns UINT32 indices, but torch doesn't have unsigned
    // 32 bit integer tensor types so we need to cast back to INT32
    indices = createCast(graph, indices, c10::ScalarType::Int)->output();

    if (negate) {
      values = createNeg(graph, {values})->output();
    }

    if (!keepdim) {
      // Squeeze out the singleton-dim left by topk
      values = createSqueeze(graph, {values}, {dim})->output();
      indices = createSqueeze(graph, {indices}, {dim})->output();
    }
  }
  replaceOutputUse(node->output(0), values);
  replaceOutputUse(node->output(1), indices);

  markNodeForDeletion(node);
  return nullptr;
}

template <typename ReduceFunc, typename ExtremaFunc>
torch::jit::Node *minMaxHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node, ReduceFunc &&reduceFunc,
                                ExtremaFunc &&extremaFunc) {
  if (node->inputs().size() == 1) {
    auto *x = node->input(0);
    auto t0 = reduceHelperDimensionCreator(x);
    return reduceFunc(graph, {x}, t0, 0);
  }
  if (node->inputs().size() == 2) {
    auto *i0 = node->input(0);
    auto *i1 = node->input(1);
    return extremaFunc(graph, {i0, i1});
  }

  return minMaxWithIndicesHandler(graph, node);
}

torch::jit::Node *minHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  return minMaxHandler(graph, node, createReducemin, createMin);
}

torch::jit::Node *maxHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  return minMaxHandler(graph, node, createReducemax, createMax);
}

torch::jit::Node *tensorNormHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  // aten::norm(Tensor in, int p) -> Tensor
  // aten::norm(Tensor in, float p) -> Tensor
  // aten::norm(Tensor in, int p, int[] dim, int keepdim) -> Tensor
  // aten::norm(Tensor in, float p, int[] dim, int keepdim) -> Tensor
  torch::jit::Value *input = node->input(0);
  torch::jit::Value *p_val = node->input(1);

  std::vector<std::int64_t> axes{};
  std::int64_t keepdim = 0;

  if (node->inputs().size() == 2) {
    torch::jit::Node *flatten = createFlatten(graph, {input}, 0);
    input = flatten->output();
    axes = {1};
  } else {
    axes = constantToLongVec(node->input(2)->node());
    keepdim = constantToLong(node->input(3)->node());
    auto shape = shapeFromTensor(input);
    // Empty axes array means reduce over all axes in PyTorch, but means
    // do nothing in PopART
    if (axes.empty()) {
      axes.resize(shape.size());
      std::iota(std::begin(axes), std::end(axes), 0);
    }
    // If we're reducing over singleton dims and keeping them, the
    // behaviour of PopART reduce ops is to do nothing, but PyTorch will
    // still take the absolute value of the tensor, so we need to
    // do the same
    if ((keepdim != 0) &&
        std::all_of(axes.begin(), axes.end(),
                    [&](std::int64_t i) { return shape[i] == 1; })) {
      return createAbs(graph, {input});
    }
  }

  constexpr float pos_inf = std::numeric_limits<float>::infinity();
  constexpr float neg_inf = -std::numeric_limits<float>::infinity();
  const float p = constantToFloat(node->input(1)->node());

  if (p == 1.0) {
    return createReducel1(graph, {input}, axes, keepdim);
  }
  if (p == 2.0) {
    return createReducel2(graph, {input}, axes, keepdim);
  }
  if (p == pos_inf || p == neg_inf) {
    // max/min(abs(x))
    torch::jit::Node *abs = createAbs(graph, {input});
    input = abs->output();

    if (p == pos_inf) {
      return createReducemax(graph, {input}, axes, keepdim);
    }
    return createReducemin(graph, {input}, axes, keepdim);
  }

  // sum(abs(x)**p)**(1./p)
  torch::jit::Node *abs = createAbs(graph, {input});

  torch::jit::Node *pow = createPow(graph, {abs->output(), p_val});
  torch::jit::Node *sum =
      createReducesum(graph, {pow->output()}, axes, keepdim);

  at::ScalarType const p_type = getNodeScalarType(p_val);

  if (p_type == c10::ScalarType::Int || p_type == c10::ScalarType::Long) {
    // Cast int to float before reciprocal
    torch::jit::Node *to_float = createCast(graph, p_val, c10::kFloat);
    p_val = to_float->output();
  }

  torch::jit::Node *one_over_p = createReciprocal(graph, {p_val});
  return createPow(graph, {sum->output(), one_over_p->output()});
}

torch::jit::Node *frobeniusnormHandler(torch::jit::Graph *graph,
                                       torch::jit::Node *node) {
  if (node->inputs().size() == 1) {
    auto *x = node->input(0);
    auto t0 = reduceHelperDimensionCreator(x);
    return createReducel2(graph, {x}, t0, 0);
  }
  if (node->inputs().size() == 3) {
    auto *x = node->input(0);
    auto *l = node->input(1);
    auto t0 = constantToLongVec(l->node());
    auto t1 = reduceHelperDimensionCreator(x, t0);
    auto *c = node->input(2);
    auto t2 = constantToLong(c->node());
    auto shape = shapeFromTensor(x);
    // If we're reducing over singleton dims and keeping them, the
    // behaviour of PopART reduce ops is to do nothing, but PyTorch will
    // still take the absolute value of the tensor, so we need to
    // do the same
    if ((t2 != 0) && std::all_of(t1.begin(), t1.end(), [&](std::int64_t i) {
          return shape[i] == 1;
        })) {
      return createAbs(graph, {x});
    }
    return createReducel2(graph, {x}, t1, t2);
  }

  ERROR("Incorrect number of arguments for operator "
        << "c10::aten::frobenius_norm. "
        << "Expecting 1 or 3 operands, "
        << "got " << node->inputs().size() << " operand(s).");
  return nullptr;
}

// count_nonzero.dim_IntList(Tensor self, int[] dim) -> Tensor
torch::jit::Node *countNonzeroHandler(torch::jit::Graph *graph,
                                      torch::jit::Node *node) {
  auto *self = node->input(0);
  auto dim = constantToLongVec(node->input(1)->node());
  if (dim.empty()) {
    dim = shapeFromTensor(self);
    std::iota(dim.begin(), dim.end(), 0);
  }

  auto *self_bool = self;
  if (getNodeScalarType(self) != c10::ScalarType::Bool) {
    self_bool = createCast(graph, self, c10::ScalarType::Bool)->output();
  }
  auto *where = createWhere(graph, {self_bool, wrapInConstant1D(graph, 1),
                                    wrapInConstant1D(graph, 0)});

  return createReducesum(graph, {where->output()}, dim, /*keepdims=*/0);
}

torch::jit::Node *nanSumHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // isNan -> where -> sum -> cast (if applicable) -> out
  torch::jit::Value *in_tensor = node->input(0);

  auto *is_nan = createIsnan(graph, {in_tensor});
  auto *zeros = createConstantFloatLike(graph, in_tensor, {0},
                                        shapeFromTensor(in_tensor));
  auto *non_nans =
      createWhere(graph, {is_nan->output(0), zeros->output(0), in_tensor});

  std::vector<int64_t> dims;
  auto *dim = node->input(1);
  if (auto *n = dim->node(); n->kind() == c10::prim::ListConstruct) {
    dims = constantToLongVec(n);
  } else if (isNone(dim)) {
    // We only get a node with Constant kind if `dim` is not
    // provided, so preform the sum over all the dimensions.
    auto in_dim_count = shapeFromTensor(in_tensor).size();
    dims.resize(in_dim_count);
    std::iota(dims.begin(), dims.end(), 0);
  } else {
    ERROR("Popart Canonicalisation: UNREACHABLE reached in nansum handler.");
  }

  auto keepdim = constantToLong(node->input(2)->node());
  auto *sum = createReducesum(graph, {non_nans->output(0)}, dims, keepdim);

  auto *dtype = node->input(3);
  if (!isNone(dtype)) {
    auto type = constantToScalarType(dtype->node());
    return createCast(graph, sum->output(0), type);
  }
  return sum;
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::amax, aMinMaxHandler);
  registerHandler(c10::aten::amin, aMinMaxHandler);
  registerHandler(c10::aten::argmax, argMinMaxHandler);
  registerHandler(c10::aten::argmin, argMinMaxHandler);
  registerHandler(c10::aten::argsort, argsortHandler);
  registerHandler(c10::aten::prod, reduceHandler);
  registerHandler(c10::aten::mean, reduceHandler);
  registerHandler(c10::aten::median, reduceMedianHandler);
  registerHandler(c10::aten::sum, reduceHandler);
  registerHandler(c10::aten::logsumexp, reduceHandler);
  registerHandler(c10::aten::norm, tensorNormHandler);
  registerHandler(c10::aten::frobenius_norm, frobeniusnormHandler);
  registerHandler(c10::aten::min, minHandler);
  registerHandler(c10::aten::minimum, minHandler);
  registerHandler(c10::aten::max, maxHandler);
  registerHandler(c10::aten::maximum, maxHandler);
  registerHandler(c10::aten::any, reduceHandler);
  registerHandler(c10::aten::all, reduceHandler);
  registerHandler(c10::aten::count_nonzero, countNonzeroHandler);
  registerHandler(c10::aten::nansum, nanSumHandler);
}
} // namespace poptorch
