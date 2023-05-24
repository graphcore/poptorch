// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"
#include "ScatterReduction.hpp"

#include "poptorch/DispatchTracer.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "../PoptorchSymbols.hpp"

#include <ATen/ATen.h>

namespace poptorch {
namespace {
torch::jit::Node *sizeHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  //  aten::size(Tensor input, int dim) -> int
  std::vector<std::int64_t> shape = shapeFromTensor(node->input(0));
  std::int64_t const dim = constantToLong(node->input(1)->node());
  return createConstantInt(graph, {shape[dim]}, {1});
}

torch::jit::Node *numToTensorHandler(torch::jit::Graph *graph,
                                     torch::jit::Node *node) {
  // Should be a tensor already
  ERROR_ON(node->input(0)->node()->kind() !=
           symbols::poptorch::tensor_constant);
  UNUSED(graph);
  node->output()->replaceAllUsesWith(node->input(0));
  markNodeForDeletion(node);
  return nullptr;
}

torch::jit::Node *flipHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // aten::flip(Tensor self, int[] dims) -> Tensor
  auto *input = node->input(0);
  // Use output shape because input shape might not exist
  // if the input is the result of another operation
  auto input_shape = shapeFromTensor(node->output());
  auto dims = constantToLongVec(node->input(1)->node());
  for (auto &dim : dims) {
    if (dim < 0) {
      dim += input_shape.size();
    }
  }
  return createReverse(graph, {input}, dims);
}

// Input tensor of shape [M, N, ...] is repeated in [R1, R2, ...]
// dimensions by:
//   1) transforming to [1, M, 1, N, ...]
//   2) expanding to [R1, M, R2, N, ...]
//   3) reshaping to [R1*M, R2*N, ...]
torch::jit::Node *repeatHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  torch::jit::Value *input = node->input(0);
  std::vector<std::int64_t> dim_repeats =
      constantToLongVec(node->input(1)->node());
  std::vector<std::int64_t> old_shape = shapeFromTensor(input);
  const std::vector<std::int64_t> new_shape = shapeFromTensor(node->output());

  // If repeat dimensions exceed shape dimensions, pad the front of the
  // original shape with singleton dimensions so that it can
  // be expanded

  std::size_t const padding = dim_repeats.size() > old_shape.size()
                                  ? dim_repeats.size() - old_shape.size()
                                  : 0;

  std::vector<std::int64_t> dim_expands;
  std::vector<std::int64_t> transform_shape;

  for (std::size_t i = 0; i < dim_repeats.size(); i++) {
    dim_expands.push_back(dim_repeats[i]);

    std::int64_t const padded_dim = i < padding ? 1 : old_shape[i - padding];
    if (padded_dim > 1 && dim_repeats[i] > 1) {
      transform_shape.push_back(1);
      dim_expands.push_back(padded_dim);
    }
    transform_shape.push_back(padded_dim);
  }

  auto *reshape = createReshape(graph, input, transform_shape);
  auto *expand = createExpand(
      graph, {reshape->output(), intVectorToIrConstant(graph, dim_expands)});

  return createReshape(graph, expand->output(), new_shape);
}

torch::jit::Node *rollHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor
  auto *input = node->input(0);
  auto input_shape = shapeFromTensor(input);
  auto shifts = constantToLongVec(node->input(1)->node());
  auto dims = constantToLongVec(node->input(2)->node());

  bool reshape_output = false;
  if (dims.empty()) {
    // If dims not provided, a flattened version of the tensor is rolled and
    // then reshaped back.
    ERROR_ON_MSG(shifts.size() != 1,
                 "The 'shifts' argument of the roll op must be a scalar when "
                 "'dims' is not specified.");
    input = createFlatten(graph, {input}, 0)->output();
    const int64_t flattened_size = std::accumulate(
        input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
    input_shape.clear();
    input_shape.push_back(1);
    input_shape.push_back(flattened_size);
    dims.push_back(1);
    reshape_output = true;
  } else {
    ERROR_ON_MSG(shifts.size() != dims.size(),
                 "The 'shifts' and 'dims' arguments of the roll op must be the "
                 "same size.");
  }

  torch::jit::Value *output = input;
  auto number_of_dims = input_shape.size();
  for (size_t i = 0; i < dims.size(); ++i) {
    auto current_dim = dims.at(i);
    // Match the torch API of requiring dim in [-len(shape), len(shape)-1]
    ERROR_ON_MSG(
        ((static_cast<std::size_t>(current_dim) >= number_of_dims) &&
         (current_dim >= 0)) ||
            ((static_cast<std::size_t>(-current_dim) > number_of_dims) &&
             (current_dim < 0)),
        "Dimension out of range at index "
            << i << " (expected to be in range of ["
            << -static_cast<std::int64_t>(number_of_dims) << ", "
            << number_of_dims - 1 << "], but got " << current_dim
            << ") in the roll op.");

    current_dim = (current_dim + number_of_dims) % number_of_dims;

    auto current_dim_size = input_shape.at(current_dim);
    // Handle overreaching and negative shifts.
    auto split = (((-shifts.at(i)) % current_dim_size) + current_dim_size) %
                 current_dim_size;
    auto *chunks = createSplit(graph, {output}, 2, current_dim,
                               {split, current_dim_size - split});
    output =
        createConcat(graph, {chunks->output(1), chunks->output(0)}, current_dim)
            ->output();
  }

  if (reshape_output) {
    return createReshape(graph, output, shapeFromTensor(node->input(0)));
  }
  return output->node();
}

torch::jit::Node *cloneHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor

  // Identity will just create a copy
  return createIdentity(graph, {node->input(0)});
}

torch::jit::Node *copyHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // aten::copy_(Tensor self, Tensor src, bool non_blocking) -> Tensor
  auto *dest = node->input(0);
  auto *src = node->input(1);
  at::ScalarType const dest_type = getNodeScalarType(dest);
  at::ScalarType const src_type = getNodeScalarType(src);

  torch::jit::Node *copy = nullptr;

  if (src_type == dest_type) {
    copy = createIdentity(graph, {src});
  } else {
    copy = createCast(graph, src, dest_type);
  }
  ERROR_ON(copy == nullptr);

  copy->output()->setType(
      copy->output()->type()->expect<c10::TensorType>()->withRequiresGrad(
          src->type()->expect<c10::TensorType>()->requiresGrad()));

  return copy;
}

torch::jit::Node *justReturnFalse(torch::jit::Graph *graph,
                                  torch::jit::Node * /*unused*/) {
  c10::IValue const value{false};
  torch::jit::Value *val = insertConstant(graph, value);
  return val->node();
}

torch::jit::Node *linearHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::linear(Tensor input, Tensor weight, Tensor? bias) -> Tensor
  auto *x = node->input(0);
  auto *w = node->input(1);
  auto *b = node->input(2);

  auto *w_t = createTranspose(graph, {w}, {1, 0});
  auto *output = createMatmul(graph, {x, w_t->output()});

  if (!isNone(b)) {
    output = createAdd(graph, {output->output(), b});
  }
  return output;
}

torch::jit::Node *gatherHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  auto *input = node->input(0);
  auto tensor_type = input->type()->expect<c10::TensorType>();
  auto axis = handleDimensionParam(node->input(1), tensor_type);
  auto *indices = node->input(2);
  auto input_shape = shapeFromTensor(input);
  auto index_shape = shapeFromTensor(indices);
  auto stride = input_shape[axis];

  for (unsigned s = 0; s < input_shape.size(); ++s) {
    if (s != axis) {
      ERROR_ON(input_shape[s] < index_shape[s]);
    }
  }

  // Move gather axis to the innermost dim
  std::vector<int64_t> permutation;
  const unsigned input_num_dims = input_shape.size();
  permutation.resize(input_num_dims);
  std::iota(permutation.begin(), permutation.end(), 0);
  permutation.push_back(permutation[axis]);
  permutation.erase(permutation.begin() + axis);

  if (axis != input_num_dims - 1) {
    input = createTranspose(graph, {input}, permutation)->output();
    input_shape.push_back(input_shape[axis]);
    input_shape.erase(input_shape.begin() + axis);
  }
  // Flatten the data
  auto *flatten_input = createFlatten(graph, {input}, 0)->output();
  int64_t num_offsets = std::accumulate(index_shape.begin(), index_shape.end(),
                                        1, std::multiplies<int64_t>());
  num_offsets /= index_shape[axis];

  // Transpose the indices to make them broadcastable with offsets
  std::vector<int64_t> idx_permutation;
  idx_permutation.resize(index_shape.size());
  std::iota(idx_permutation.begin(), idx_permutation.end(), 0);
  idx_permutation.insert(idx_permutation.begin(), idx_permutation[axis]);
  idx_permutation.erase(idx_permutation.begin() + axis + 1);

  if (axis != 0) {
    indices = createTranspose(graph, {indices}, idx_permutation)->output();
    index_shape.insert(index_shape.begin(), index_shape[axis]);
    index_shape.erase(index_shape.begin() + (axis + 1));
  }
  // Create shape for offsets that is broadcastable with indices tensor
  std::vector<int64_t> offset_shape = {index_shape.begin() + 1,
                                       index_shape.end()};
  // Make the offsets
  std::vector<int64_t> offsets_val;
  int64_t num_data = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                     std::multiplies<int64_t>());
  num_data /= input_shape[input_num_dims - 1];
  torch::jit::Value *offsets;

  // Case where one or more indices dims size < data size
  if (num_offsets != num_data) {
    // Create the offsets tensor from data_size
    // then slice it to match indices_size
    auto data_shape = shapeFromTensor(node->input(0));
    data_shape.insert(data_shape.begin(), data_shape[axis]);
    data_shape.erase(data_shape.begin() + (axis + 1));
    std::vector<int64_t> temp_offsets_shape = {data_shape.begin() + 1,
                                               data_shape.end()};
    offsets_val.resize(num_data);
    std::iota(offsets_val.begin(), offsets_val.end(), 0);

    for (auto &v : offsets_val) {
      v *= stride;
    }
    offsets =
        createConstantInt(graph, offsets_val, temp_offsets_shape)->output();

    for (unsigned k = 0; k < offset_shape.size(); ++k) {
      if (offset_shape[k] != temp_offsets_shape[k]) {
        offsets = createSlice(graph, {offsets}, {offset_shape[k]}, {0}, {k})
                      ->output();
      }
    }
  } else {
    offsets_val.resize(num_offsets);
    std::iota(offsets_val.begin(), offsets_val.end(), 0);

    for (auto &v : offsets_val) {
      v *= stride;
    }
    offsets = createConstantInt(graph, offsets_val, {offset_shape})->output();
  }

  auto *new_indices = createAdd(graph, {indices, offsets})->output();
  // Gather the elements
  auto *output = createGather(graph, {flatten_input, new_indices}, 1)->output();
  // remove the dim-0 added by gather
  output = createSqueeze(graph, {output}, {0})->output();
  // transpose back to the original indices shape if needed
  if (axis != 0) {
    std::iota(idx_permutation.begin(), idx_permutation.end(), 0);
    idx_permutation.erase(idx_permutation.begin());
    idx_permutation.insert(idx_permutation.begin() + axis, 0);
    output = createTranspose(graph, {output}, idx_permutation)->output();
  }
  return output->node();
}

torch::jit::Node *takeAlongDimHandler(torch::jit::Graph *graph,
                                      torch::jit::Node *node) {
  // aten::take_along_dim(Tensor self, Tensor indices, int? dim=None) -> Tensor

  torch::jit::Value *input = node->input(0);
  torch::jit::Value *indices = node->input(1);
  torch::jit::Value *dim = node->input(2);

  const std::vector<std::int64_t> input_shape = shapeFromTensor(input);
  std::vector<std::int64_t> indices_shape = shapeFromTensor(indices);

  if (!isNone(dim)) {
    const auto dim_value = constantToLong(dim->node());

    const auto broadcast_to = [&](torch::jit::Value *value,
                                  const std::vector<std::int64_t> &shape) {
      std::vector<torch::jit::Value *> shape_values(shape.size(), nullptr);
      std::transform(shape.cbegin(), shape.cend(), shape_values.begin(),
                     [&](const auto elem) -> torch::jit::Value * {
                       return wrapInConstant1D(graph, elem);
                     });

      torch::jit::Value *shape_list =
          createAndInsertNode(graph, c10::prim::ListConstruct, shape_values)
              ->output();

      auto *broadcasted_value =
          createHandlerOperation(graph, getHandler(c10::aten::broadcast_to),
                                 {value, shape_list})
              ->output();

      broadcasted_value->setType(
          value->type()->expect<c10::TensorType>()->withSizes(shape));
      return broadcasted_value;
    };

    auto self_sizes = input_shape;
    // update number of elements at dim as per indices
    self_sizes.at(dim_value) = indices_shape.at(dim_value);
    if (auto bcast_shape = at::infer_size(self_sizes, indices_shape);
        bcast_shape != indices_shape) {
      indices = broadcast_to(indices, bcast_shape);
    }

    // update number of elements at dim as per self
    indices_shape.at(dim_value) = input_shape.at(dim_value);
    if (auto bcast_shape = at::infer_size(indices_shape, input_shape);
        bcast_shape != input_shape) {
      input = broadcast_to(input, bcast_shape);
    }
  } else {
    const auto flatten =
        [&](torch::jit::Value *value,
            const std::vector<std::int64_t> &shape) -> torch::jit::Value * {
      const auto rank = shape.size();
      if (rank == 1) {
        return value;
      }

      const int64_t num_elems =
          rank > 1 ? std::accumulate(shape.cbegin(), shape.cend(), 1,
                                     std::multiplies<int64_t>())
                   : 1;

      return createReshape(graph, value, {num_elems})->output();
    };

    input = flatten(input, input_shape);
    indices = flatten(indices, indices_shape);
    dim = wrapInConstant1D(graph, 0);
  }

  return createHandlerOperation(graph, getHandler(c10::aten::gather),
                                {input, dim, indices});
}

torch::jit::Node *scatterHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  auto *input = node->input(0);
  const auto input_type = input->type()->expect<c10::TensorType>();
  const auto dim = handleDimensionParam(node->input(1), input_type);
  auto *index = node->input(2);
  auto *src = node->input(3);

  // `scatter` can be passed a single value for `src` as a tensor constant, so
  // broadcast it up.
  if (isConstantScalar(src)) {
    auto *shape = intVectorToIrConstant(graph, shapeFromTensor(index));
    const auto input_scalar_type = *input_type->scalarType();
    if (input_scalar_type !=
        *src->type()->expect<c10::TensorType>()->scalarType()) {
      // poplibs scatter requires that `src` have the same data type as input so
      // cast it if needed
      src = castToPromoteType(graph, src, input_scalar_type);
    }
    src = createExpand(graph, {src, shape})->output();
  }

  if (node->inputs().size() < 4) {
    return createScatterElements(graph, {input, index, src}, dim);
  }

  // reduction type is optional argument
  const auto reduce = node->inputs().size() < 5
                          ? static_cast<std::int32_t>(ScatterReduction::None)
                          : getReductionMethod(node->input(4)->node());
  const auto input_shape = shapeFromTensor(input);
  const auto axis_size = input_shape.at(dim);
  static constexpr bool enable_index_broadcast = false;

  return createScatterreduce(graph, {src, index, input}, axis_size, dim,
                             enable_index_broadcast, reduce);
}

torch::jit::Node *fullCommon(torch::jit::Graph *graph, torch::jit::Value *v,
                             at::ScalarType type,
                             const std::vector<int64_t> &shape) {
  auto *vn = v->node();
  auto stype = coerceToSupportedType(type);
  if (isTensorConstant(vn) && vn->output()->type()->cast<c10::TensorType>()) {
    auto v_scalar = getNodeTensorAttrValue(vn).to(stype).item();
    return tensorToConstant(graph, at::full(shape, v_scalar, stype));
  }
  auto *v_cast = createCast(graph, v, stype)->output();
  auto *c_shape = intVectorToIrConstant(graph, shape);
  return createExpand(graph, {v_cast, c_shape});
}

torch::jit::Node *fullHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // aten::full(int[] size, Scalar fill_value,
  //            ScalarType? dtype=None, Layout? layout=None,
  //            Device? device=None, bool? pin_memory=None) -> Tensor
  // aten::new_full(Tensor self, int[] size, Scalar fill_value,
  //                ScalarType? dtype=None, Layout? layout=None,
  //                Device? device=None, bool? pin_memory=None) -> Tensor
  size_t shape_index = 0;
  if (node->kind() == c10::aten::new_full) {
    shape_index = 1;
  }
  auto *shape = node->input(shape_index + 0);
  auto *v = node->input(shape_index + 1);
  auto *dtype = node->input(shape_index + 2);
  auto lv_shape = constantToLongVec(shape->node());
  auto type = c10::ScalarType::Float;
  if (node->kind() == c10::aten::new_full) {
    type = getNodeScalarType(node->input(0));
  }
  // The specified dtype takes precedence
  if (!isNone(dtype)) {
    type = constantToScalarType(dtype->node());
  }

  return fullCommon(graph, v, type, lv_shape);
}

torch::jit::Node *fullLikeHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  // aten::full_like(Tensor self, Scalar fill_value) -> Tensor
  auto *v = node->input(1);
  auto *like = node->output(0);
  auto like_shape = shapeFromTensor(like);
  auto like_type = getNodeScalarType(like);
  return fullCommon(graph, v, like_type, like_shape);
}

torch::jit::Node *triuHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // aten::triu(Tensor self, int diagonal=0) -> Tensor
  ERROR("torch.triu is only supported within constant expressions, "
        "for example torch.ones(3, 3).triu_().");
  UNUSED(graph);
  UNUSED(node);
  return nullptr;
}

torch::jit::Node *ipuPrintTensorHandler(torch::jit::Graph *graph,
                                        torch::jit::Node *node) {
  auto *x = node->input(0);
  auto title = constantToString(node->input(1)->node());
  auto print_gradient = constantToInt(node->input(2)->node());
  auto summarise_threshold = constantToInt(node->input(3)->node());
  auto edge_items = constantToInt(node->input(4)->node());
  auto max_line_width = constantToInt(node->input(5)->node());
  auto digits = constantToInt(node->input(6)->node());
  auto float_format = constantToInt(node->input(7)->node());
  auto separator = constantToString(node->input(8)->node());
  auto open_bracket = constantToString(node->input(9)->node());
  auto close_bracket = constantToString(node->input(10)->node());
  return createPrinttensor(graph, {x}, print_gradient, title,
                           summarise_threshold, edge_items, max_line_width,
                           digits, float_format, *separator.c_str(),
                           *open_bracket.c_str(), *close_bracket.c_str());
}
} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::size, sizeHandler);
  registerHandler(c10::prim::NumToTensor, numToTensorHandler);
  registerHandler(c10::aten::flip, flipHandler);
  registerHandler(c10::aten::repeat, repeatHandler);
  registerHandler(c10::aten::is_complex, justReturnFalse);
  registerHandler(c10::aten::roll, rollHandler);
  registerHandler(c10::aten::clone, cloneHandler);
  registerHandler(c10::aten::copy_, copyHandler);
  registerHandler(c10::aten::linear, linearHandler);
  registerHandler(c10::aten::gather, gatherHandler);
  registerHandler(c10::aten::scatter, scatterHandler);
  registerHandler(c10::aten::full, fullHandler);
  registerHandler(c10::aten::new_full, fullHandler);
  registerHandler(c10::aten::full_like, fullLikeHandler);
  registerHandler(c10::aten::triu, triuHandler);
  registerHandler(symbols::poptorch::ipu_print_tensor, ipuPrintTensorHandler);
  registerHandler(c10::aten::take_along_dim, takeAlongDimHandler);
}

} // namespace poptorch
