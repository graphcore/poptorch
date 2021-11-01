
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "../PoptorchSymbols.hpp"

namespace poptorch {
namespace {

torch::jit::Node *expandHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::expand(Tensor self, int[] size)  -> Tensor
  torch::jit::Node *new_node;

  // Extract the type from the pytorch IR.
  c10::TensorTypePtr self_tensor =
      node->input(0)->type()->expect<c10::TensorType>();
  c10::VaryingShape self_dims = self_tensor->sizes();

  // Old shape
  std::vector<std::int64_t> old_shape = shapeFromTensor(node->input(0));

  // Count the elems in the old shape.
  std::int64_t old_elem_count = std::accumulate(
      old_shape.begin(), old_shape.end(), 1, std::multiplies<std::int64_t>());

  // Get the target size for the expand.
  std::vector<std::int64_t> new_shape =
      constantToLongVec(node->input(1)->node());

  // a new shape element of -1 means that dimension should not change
  for (unsigned i = 0; i < new_shape.size(); ++i) {
    if (new_shape[i] == -1) {
      new_shape[i] = old_shape[i];
    }
  }

  // Count the number of elements in the target shape.
  std::int64_t new_elem_count = std::accumulate(
      new_shape.begin(), new_shape.end(), 1, std::multiplies<std::int64_t>());

  // Elements don't change so just a reshape.
  if (new_elem_count == old_elem_count) {
    new_node = createReshape(graph, node->input(0), new_shape);
  } else {
    // Otherwise we are expanding the original tensor.
    new_node = createConstantInt(graph, new_shape,
                                 {static_cast<int64_t>(new_shape.size())});
    new_node = createCast(graph, new_node->output(), c10::kLong);
    new_node = createExpand(graph, {node->input(0), new_node->output()});
  }
  return new_node;
}

torch::jit::Node *flattenHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  // flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) ->
  // Tensor(a)

  std::int64_t start_dim = constantToLong(node->input(1)->node());
  std::int64_t end_dim = constantToLong(node->input(2)->node());

  c10::TensorTypePtr self_tensor =
      node->input(0)->type()->expect<c10::TensorType>();
  c10::VaryingShape self_dims = self_tensor->sizes();

  // Respect PyTorch negative dimensions
  if (end_dim < 0) {
    end_dim = (*self_dims.sizes()).size() + end_dim;
  }

  if (start_dim < 0) {
    start_dim = (*self_dims.sizes()).size() + start_dim;
  }

  std::vector<std::int64_t> new_shape;

  int dim = 0;
  std::int64_t flattened_dims = 1;

  // Flatten the selected dimensions.
  for (auto optional_int : *self_dims.sizes()) {
    if (dim < start_dim || dim > end_dim) {
      new_shape.push_back(*optional_int);
    } else {
      flattened_dims *= *optional_int;
    }

    if (dim == end_dim) {
      new_shape.push_back(flattened_dims);
    }

    dim++;
  }

  return createReshape(graph, node->input(0), new_shape);
}

torch::jit::Node *reshapeHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  // aten::view(Tensor self, int[] size) -> Tensor
  // aten::unsqueeze(Tensor self, int dim) -> Tensor
  // aten::view(Tensor(a) self, int[] size) -> (Tensor(a))
  std::vector<std::int64_t> new_shape = shapeFromTensor(node->output());

  // Reshape the tensor into that shape.
  return createReshape(graph, node->input(0), new_shape);
}

torch::jit::Node *expandAsHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  // aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor
  // aten::expand_as(Tensor self, Tensor other) -> Tensor
  torch::jit::Node *new_node;

  // Extract the type from the pytorch IR.
  c10::TensorTypePtr self_tensor =
      node->input(0)->type()->expect<c10::TensorType>();
  c10::VaryingShape self_dims = self_tensor->sizes();

  std::int64_t old_elem_count = 0;
  for (auto optional_int : *self_dims.sizes()) {
    old_elem_count += *optional_int;
  }

  // Extract the type from the pytorch IR.
  c10::TensorTypePtr as_tensor =
      node->input(1)->type()->expect<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  // Convert that IR type into a C++ vector of ints.
  std::vector<std::int64_t> new_shape;
  std::int64_t new_elem_count = 0;

  for (auto optional_int : *dims.sizes()) {
    new_shape.push_back(*optional_int);
    new_elem_count += *optional_int;
  }

  // Elements don't change so just a reshape.
  if (new_elem_count == old_elem_count) {
    new_node = createReshape(graph, node->input(0), new_shape);
  } else {
    new_node = createConstantInt(graph, new_shape,
                                 {static_cast<int64_t>(new_shape.size())});

    new_node = createCast(graph, new_node->output(), c10::kLong);

    new_node = createExpand(graph, {node->input(0), new_node->output()});
  }
  return new_node;
}

torch::jit::Node *selectHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::select(Tensor self, int dim, int index) -> Tensor

  // Note: there is also this overload which is not supported at the moment
  // aten::select(Tensor[] list, int idx) -> Tensor
  std::int64_t dim = constantToLong(node->input(1)->node());

  auto *index_node = node->input(2)->node();

  torch::jit::Node *slice_node;
  if (!isTensorConstant(index_node)) {
    // Handle dynamic index
    slice_node = createDynamicslice(
        graph, {node->input(0), index_node->output()}, {dim}, {1}, 1);
  } else {
    // Handle static index
    std::int64_t index = constantToLong(index_node);

    if (index < 0) {
      c10::TensorTypePtr as_tensor =
          node->input(0)->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = as_tensor->sizes();
      index += *dims[dim];
    }

    slice_node =
        createSlice(graph, {node->input(0), wrapInConstant1D(graph, index),
                            wrapInConstant1D(graph, index + 1),
                            wrapInConstant1D(graph, dim)});
  }

  // Reshape to remove the singleton dimenson left in by slice
  auto original_shape = shapeFromTensor(node->output());
  return createReshape(graph, slice_node->output(), original_shape);
}

torch::jit::Node *contiguousHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  // aten::contiguous(Tensor self, *, MemoryFormat
  // memory_format=contiguous_format) -> Tensor Returns a copy of the tensor but
  // in contiguous memory.
  //
  // Returns the tensor
  UNUSED(graph);
  node->output()->replaceAllUsesWith(node->input(0));
  markNodeForDeletion(node);
  return nullptr;
}

torch::jit::Node *permuteHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  // aten::permute(Tensor self, int[] dims) -> Tensor

  std::vector<std::int64_t> permutation =
      constantToLongVec(node->input(1)->node());

  c10::TensorTypePtr as_tensor =
      node->input(0)->type()->cast<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  std::for_each(permutation.begin(), permutation.end(), [&](std::int64_t &val) {
    if (val < 0) {
      val = *dims.size() + val;
    }
  });

  return createTranspose(graph, {node->input(0)}, permutation);
}

torch::jit::Node *transposeHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  // aten::transpose(Tensor self, int dim0, int dim1) -> Tensor
  std::int64_t dim0 = constantToLong(node->input(1)->node());

  std::int64_t dim1 = constantToLong(node->input(2)->node());

  c10::TensorTypePtr as_tensor =
      node->input(0)->type()->cast<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  // Convert that IR type into a C++ vector of ints. In popart the
  // permutation includes all elements (rotate last two elements with [0, 1,
  // 3, 2]) whereas in pytorch you only need to specify the dimensions being
  // moved (same operation, [3, 2]). So we need to make sure the IR reflects
  // that.
  std::vector<std::int64_t> permutation;
  for (std::uint64_t i = 0; i < *dims.size(); ++i) {
    permutation.push_back(i);
  }

  // Allow for python array style access.
  if (dim0 < 0) {
    dim0 = *dims.size() + dim0;
  }

  if (dim1 < 0) {
    dim1 = *dims.size() + dim1;
  }

  permutation[dim0] = dim1;
  permutation[dim1] = dim0;

  return createTranspose(graph, {node->input(0)}, permutation);
}

torch::jit::Node *numpyTHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  auto shape = shapeFromTensor(node->input(0));

  if (shape.size() < 2) {
    return node->input(0)->node();
  }

  std::vector<std::int64_t> permutation;
  for (std::int64_t i = shape.size() - 1; i >= 0; i--) {
    permutation.push_back(i);
  }

  return createTranspose(graph, {node->input(0)}, permutation);
}

torch::jit::Node *splitChunkHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  // aten::split(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]"
  // aten::split(Tensor self, int split_sizes, int dim=0) -> Tensor[]"
  // aten::chunk(Tensor self, int chunks, int dim) -> Tensor[]
  // aten::unsafe_chunk(Tensor self, int chunks, int dim) -> Tensor[]

  torch::jit::Symbol kind = node->kind();
  // Get the shape of the input.
  c10::TensorTypePtr as_tensor =
      node->input(0)->type()->expect<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  // Pythonic axis translation.
  const std::int64_t dim = constantToLong(node->input(2)->node());
  const std::int64_t axis = dim >= 0 ? dim : *dims.size() + dim;

  // Size of each split ignoring the remainder at the end.
  std::vector<std::int64_t> size_of_each_split;

  // Split size can either be the number of splits or the size of the
  // splits.
  std::optional<std::int64_t> split_size;

  if (node->input(1)->node()->kind() == symbols::poptorch::tensor_constant) {
    ERROR_ON(getNodeScalarType(node->input(1)) != at::ScalarType::Int);
    split_size = constantToLong(node->input(1)->node());
  }

  if (kind == c10::aten::chunk || kind == c10::aten::unsafe_chunk) {
    // Chunk takes in the *number of chunks*. Canonicalise it to *size of
    // chunks*.
    ERROR_ON_MSG(!split_size,
                 "Aten chunk node does not have a integer number of chunks!");
    std::int64_t slice_size = *dims[axis] / *split_size;
    for (int i = 0; i < *split_size; ++i) {
      size_of_each_split.push_back(slice_size);
    }

    // Add an extra slice for the remainder.
    if (*dims[axis] % *split_size != 0) {
      size_of_each_split.push_back(*dims[axis] % *split_size);
    }
  } else if (split_size) {
    // Split takes in the size of each chunk.
    std::int64_t slice_size = *split_size;
    for (int i = 0; i < *dims[axis] / slice_size; ++i) {
      size_of_each_split.push_back(slice_size);
    }

    // Add an extra slice for the remainder.
    if (*dims[axis] % *split_size != 0) {
      size_of_each_split.push_back(*dims[axis] % *split_size);
    }
  } else {
    size_of_each_split = constantToLongVec(node->input(1)->node());
  }

  // Rolling index to track where we are in the tensor.
  std::int64_t index = 0;

  // The result of each slice.
  std::vector<torch::jit::Value *> slices;

  // Slice up according to the canonicalised split vector.
  for (std::int64_t slice_size : size_of_each_split) {
    torch::jit::Node *slice =
        createSlice(graph, {node->input(0), wrapInConstant1D(graph, index),
                            wrapInConstant1D(graph, index + slice_size),
                            wrapInConstant1D(graph, axis)});

    // Add the slice to the graph.
    slices.push_back(slice->output());

    // Move along in the vector dimension.
    index += slice_size;
  }

  auto *list_node = createAndInsertNode(graph, at::prim::ListConstruct, slices);
  ERROR_ON(node->output()->uses().size() != 1);
  auto *unpack = node->output()->uses()[0].user;
  ERROR_ON(unpack->kind() != c10::prim::ListUnpack);
  ERROR_ON(slices.size() != unpack->outputs().size());

  // Propagate types
  for (size_t i = 0; i < slices.size(); i++) {
    unpack->output(i)->setType(slices[i]->type());
  }

  return list_node;
}

torch::jit::Node *toHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto tensor_type = node->input(0)->type()->cast<c10::TensorType>();
  ERROR_ON_MSG(!tensor_type,
               "Casting from a non-tensor type not supported, in an aten::to.");

  // aten::to(Tensor(a) self, Device? device, int? dtype=None, bool
  // non_blocking=False, bool copy=False) -> Tensor(a|b)" aten::to(Tensor(a)
  // self, int? dtype=None, bool non_blocking=False, bool copy=False) ->
  // Tensor(a|b)" aten::to(Tensor(a) self, [args without dtype])

  std::optional<c10::ScalarType> cast_to;
  if (node->input(1)->type()->cast<c10::DeviceObjType>() ||
      node->input(1)->type()->cast<c10::TensorType>()) {
    cast_to = getNodeScalarType(node->output(0));
  }

  if (cast_to.has_value()) {
    // In this case, the original dtype may have been half
    if (*cast_to == at::ScalarType::Float) {
      cast_to = HALF_OR_FLOAT;
    }

    // Avoid promoting to an unsupported type
    if (*cast_to == at::ScalarType::Double) {
      cast_to = at::ScalarType::Float;
    } else if (*cast_to == at::ScalarType::Long) {
      cast_to = at::ScalarType::Int;
    }
  }

  if (!cast_to.has_value() || cast_to == *tensor_type->scalarType()) {
    // NOOP
    if (cast_to == *tensor_type->scalarType()) {
      logging::trace("Ignoring type cast to same type, {}, {}", *cast_to,
                     *tensor_type->scalarType());
    }

    node->output()->replaceAllUsesWith(node->input(0));
    markNodeForDeletion(node);
    return nullptr;
  }
  return createCast(graph, node->input(0), *cast_to);
}

torch::jit::Node *upsampleHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  // aten::upsample_nearest1d(Tensor self, int[] output_size, float? scales) ->
  // Tensor
  //
  // aten::upsample_nearest2d(Tensor self, int[] output_size, float?
  // scales_h, float? scales_w) -> Tensor
  //
  // aten::upsample_nearest3d(Tensor self, int[] output_size, float? scales_d,
  // float? scales_h, float? scales_w) -> Tensor
  //
  // Not supported by Popart yet:
  //
  // aten::upsample_linear1d(Tensor self, int[] output_size, bool align_corners,
  // float? scales) -> Tensor
  //
  // aten::upsample_trilinear3d(Tensor self, int[] output_size, bool
  // align_corners, float? scales_d, float? scales_h, float? scales_w) -> Tensor

  torch::jit::Value *input = node->input(0);
  torch::jit::Value *output_size = node->input(1);
  torch::jit::Value *output_scale = node->input(2);

  auto output_rank = shapeFromTensor(node->output()).size();
  auto input_shape = shapeFromTensor(input);
  auto input_rank = input_shape.size();

  ERROR_ON_MSG(output_rank != input_rank,
               "Input / output rank mismatch: " << input_rank
                                                << " != " << output_rank);

  // Omit the leading batch and channel dims for computing the scale
  std::vector<double> scales{1.0, 1.0};

  if (!isNone(output_size)) {
    auto output_shape = handleTensorList(output_size->node());
    for (size_t dim = 2; dim < input_rank; ++dim) {
      scales.push_back(constantToFloat(output_shape[dim - 2]->node()) /
                       input_shape[dim]);
    }
  } else {
    for (auto *s : handleTensorList(output_scale->node())) {
      scales.push_back(constantToFloat(s->node()));
    }
  }

  torch::jit::Node *scales_node = createConstantFloatLike(
      graph, input, scales, {static_cast<std::int64_t>(scales.size())});
  return createResize(graph, {input, scales_node->output()}, "nearest");
}

torch::jit::Node *upsampleBilinear2dHandler(torch::jit::Graph *graph,
                                            torch::jit::Node *node) {
  auto *input = node->input(0);
  auto *output_size = node->input(1);
  auto *output_scale = node->input(3);

  auto scalar_type = getNodeScalarType(input);
  auto output_rank = shapeFromTensor(node->output()).size();
  auto input_shape = shapeFromTensor(input);
  auto input_rank = input_shape.size();

  ERROR_ON_MSG(output_rank != input_rank,
               "Input / output rank mismatch: " << input_rank
                                                << " != " << output_rank);

  // Omit the leading batch and channel dims for computing the scale
  std::vector<double> scales{1.0, 1.0};

  if (!isNone(output_size)) {
    auto output_shape = constantToLongVec(output_size->node());
    for (size_t dim = 2; dim < input_rank; ++dim) {
      scales.push_back(static_cast<double>(output_shape[dim - 2]) /
                       input_shape[dim]);
    }
  } else {
    auto scalesxy = constantToFloatVec(output_scale->node());

    ERROR_ON_MSG(scalesxy[0] != scalesxy[1],
                 "Non-uniform bilinear upsampling not supported");
    ERROR_ON_MSG(scalesxy[0] != floor(scalesxy[0]),
                 "Bilinear upsampling with non-integer factor not supported");

    scales.push_back(scalesxy[0]);
    scales.push_back(scalesxy[1]);
  }

  std::vector<torch::jit::Value *> inputs = {input};
  std::string name = "UpsampleBilinear2d";
  std::string domain = "poptorch.custom_ops";
  std::string attributes("{\"scaling_factor\":" + std::to_string(scales[2]) +
                         "}");

  auto *new_node =
      createCustomOperation(graph, inputs, name, domain, 1, 1, attributes);
  new_node->output(0)->setType(c10::TensorType::create(
      scalar_type, c10::nullopt, c10::nullopt, c10::nullopt));
  return new_node;
}

torch::jit::Node *unsupportedUpsampleHandler(torch::jit::Graph *graph,
                                             torch::jit::Node *node) {
  UNUSED(graph);
  ERROR("Unsupported upsample mode "
        << node->kind().toQualString()
        << ": currently only 'nearest' is supported");
  return nullptr;
}

torch::jit::Node *stackHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  std::int64_t dim = constantToLong(node->input(1)->node());

  std::vector<torch::jit::Value *> values =
      handleTensorList(node->input(0)->node());

  std::vector<torch::jit::Value *> transformed_tensors;

  transformed_tensors.reserve(values.size());
  for (auto *value : values) {
    transformed_tensors.push_back(
        createUnsqueeze(graph, {value}, {dim})->output());
  }

  return createConcat(graph, transformed_tensors, dim);
}

torch::jit::Node *intHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  return createCast(graph, node->input(0), at::ScalarType::Int);
}

torch::jit::Node *autocastHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  auto from_type = getNodeScalarType(node->input(0));
  auto to_type = getNodeScalarType(node->output(0));

  if (from_type == to_type) {
    node->output()->replaceAllUsesWith(node->input(0));
    markNodeForDeletion(node);
    return nullptr;
  }

  return createCast(graph, node->input(0), to_type);
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::expand, expandHandler);
  registerHandler(c10::aten::expand_as, expandAsHandler);
  registerHandler(c10::aten::view, reshapeHandler);
  registerHandler(c10::aten::unsqueeze, reshapeHandler);
  registerHandler(c10::aten::flatten, flattenHandler);
  registerHandler(c10::aten::reshape, reshapeHandler);
  registerHandler(c10::aten::_reshape_alias, reshapeHandler);
  registerHandler(c10::aten::select, selectHandler);
  registerHandler(c10::aten::split, splitChunkHandler);
  registerHandler(c10::aten::split_with_sizes, splitChunkHandler);
  registerHandler(c10::aten::chunk, splitChunkHandler);
  registerHandler(c10::aten::unsafe_chunk, splitChunkHandler);
  registerHandler(c10::aten::contiguous, contiguousHandler);
  registerHandler(c10::aten::permute, permuteHandler);
  registerHandler(c10::aten::transpose, transposeHandler);
  registerHandler(c10::aten::numpy_T, numpyTHandler);
  registerHandler(c10::aten::to, toHandler);
  registerHandler(c10::aten::type_as, toHandler);
  registerHandler(c10::aten::upsample_nearest1d, upsampleHandler);
  registerHandler(c10::aten::upsample_nearest2d, upsampleHandler);
  registerHandler(c10::aten::upsample_nearest3d, upsampleHandler);
  registerHandler(c10::aten::upsample_linear1d, unsupportedUpsampleHandler);
  registerHandler(c10::aten::upsample_bilinear2d, upsampleBilinear2dHandler);
  registerHandler(c10::aten::upsample_trilinear3d, unsupportedUpsampleHandler);
  registerHandler(c10::aten::upsample_bicubic2d, unsupportedUpsampleHandler);
  registerHandler(c10::aten::squeeze, reshapeHandler);
  registerHandler(c10::aten::as_strided, reshapeHandler);
  registerHandler(c10::aten::stack, stackHandler);
  registerHandler(c10::aten::Int, intHandler);
  registerHandler(symbols::poptorch::autocast, autocastHandler);
}

} // namespace poptorch
