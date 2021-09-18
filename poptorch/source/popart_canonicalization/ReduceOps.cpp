// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <limits>

#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

#include "../PoptorchSymbols.hpp"
#include "poptorch/Utils.hpp"

namespace poptorch {
namespace {

torch::jit::Node *reduceHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // Reductions have two overloads. The first is:
  // aten::mean(Tensor self, int[] dim, int keepdim, Tensor? out)) -> tensor

  // The second is:
  // aten::mean(Tensor self, int? dtype)) -> tensor

  torch::jit::Symbol kind = node->kind();
  torch::jit::Value *input = node->input(0);

  // sum and prod works even for bool types in PyTorch
  auto tensor_type = input->type()->expect<c10::TensorType>();
  if (tensor_type->scalarType() == at::ScalarType::Bool) {
    auto cast_node = createCast(graph, input, c10::ScalarType::Int);
    input = cast_node->output();
  }

  std::vector<std::int64_t> axes{};
  std::int64_t keepdim = 0;

  // Case 2.
  if (node->inputs().size() == 2) {
    // Need to use reshape as "Flatten" is for 2D output
    auto numels_optional = tensor_type->numel();
    ERROR_ON(!numels_optional);
    input =
        createReshape(graph, input, {static_cast<int64_t>(*numels_optional)})
            ->output();
    axes = {0};
  } else {
    // Case 1.
    // Sometimes the dimensions are just one int.

    if (node->input(1)->node()->kind() == symbols::poptorch::tensor_constant) {
      axes.push_back(constantToLong(node->input(1)->node()));
    } else {
      axes = constantToLongVec(node->input(1)->node());
    }

    keepdim = constantToLong(node->input(2)->node());
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
  ERROR("Popart Canonicalisation: UNREACHABLE reached in reductions.");
}

torch::jit::Node *reduceMedianHandler(torch::jit::Graph *graph,
                                      torch::jit::Node *node) {
  auto input = node->input(0);
  std::vector<std::int64_t> axes;
  std::int64_t keepdim = 0;

  torch::jit::Node *output;

  if (node->inputs().size() == 1) {
    // aten::median(Tensor self) -> Tensor
    axes = reduceHelperDimensionCreator(input);
    auto reduced = createReducemedian(graph, {input}, axes, keepdim);
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
  auto input = node->input(0);
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

  torch::jit::Symbol kind = node->kind();
  torch::jit::Value *input = node->input(0);

  std::optional<std::int64_t> dim;
  if (node->input(1)->node()->kind() == symbols::poptorch::tensor_constant) {
    dim = constantToLong(node->input(1)->node());
  }

  std::int64_t keep_dim = constantToLong(node->input(2)->node());

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

  // Create the actual argmax/argmin.
  if (kind == c10::aten::argmax) {
    return createArgmax(graph, {input}, dim_to_use, keep_dim);
  }
  return createArgmin(graph, {input}, dim_to_use, keep_dim);
}

torch::jit::Node *minMaxWithIndicesHandler(torch::jit::Graph *graph,
                                           torch::jit::Node *node) {
  auto x = node->input(0);
  auto t0 = x->type()->expect<c10::TensorType>();
  auto dim = handleDimensionParam(node->input(1), t0);
  auto keepdim = constantToBool(node->input(2)->node());
  bool negate = node->kind() == c10::aten::min;

  if (negate) {
    x = createNeg(graph, {x})->output();
  }

  auto one = tensorToConstant(graph, at::tensor(1L))->output();
  auto result = createTopk(graph, {x, one}, dim);
  auto values = result->output(0);
  auto indices = result->output(1);
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

  if (node->hasUses()) {
    replaceOutputUse(node->output(0), values);
    replaceOutputUse(node->output(1), indices);
  }

  markNodeForDeletion(node);
  return nullptr;
}

template <typename ReduceFunc, typename ExtremaFunc>
torch::jit::Node *minMaxHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node, ReduceFunc &&reduceFunc,
                                ExtremaFunc &&extremaFunc) {
  if (node->inputs().size() == 1) {
    auto x = node->input(0);
    auto t0 = reduceHelperDimensionCreator(x);
    return reduceFunc(graph, {x}, t0, 0);
  }
  if (node->inputs().size() == 2) {
    auto i0 = node->input(0);
    auto i1 = node->input(1);
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
  float p = constantToFloat(node->input(1)->node());

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

  at::ScalarType p_type = getNodeScalarType(p_val);

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
    auto x = node->input(0);
    auto t0 = reduceHelperDimensionCreator(x);
    return createReducel2(graph, {x}, t0, 0);
  }
  if (node->inputs().size() == 3) {
    auto x = node->input(0);
    auto l = node->input(1);
    auto t0 = constantToLongVec(l->node());
    auto t1 = reduceHelperDimensionCreator(x, t0);
    auto c = node->input(2);
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

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::amax, aMinMaxHandler);
  registerHandler(c10::aten::amin, aMinMaxHandler);
  registerHandler(c10::aten::argmax, argMinMaxHandler);
  registerHandler(c10::aten::argmin, argMinMaxHandler);
  registerHandler(c10::aten::prod, reduceHandler);
  registerHandler(c10::aten::mean, reduceHandler);
  registerHandler(c10::aten::median, reduceMedianHandler);
  registerHandler(c10::aten::sum, reduceHandler);
  registerHandler(c10::aten::logsumexp, reduceHandler);
  registerHandler(c10::aten::norm, tensorNormHandler);
  registerHandler(c10::aten::frobenius_norm, frobeniusnormHandler);
  registerHandler(c10::aten::min, minHandler);
  registerHandler(c10::aten::max, maxHandler);
}
} // namespace poptorch
