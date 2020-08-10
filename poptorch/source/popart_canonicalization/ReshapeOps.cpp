// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

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

torch::jit::Node *reshapeHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  // aten::view(Tensor self, int[] size) -> Tensor
  // aten::unsqueeze(Tensor self, int dim) -> Tensor

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

  std::int64_t index = constantToLong(node->input(2)->node());
  if (index < 0) {
    c10::TensorTypePtr as_tensor =
        node->input(0)->type()->cast<c10::TensorType>();
    c10::VaryingShape dims = as_tensor->sizes();
    index += *dims[dim];
  }

  return createSlice(graph, {node->input(0), wrapInConstant1D(graph, index),
                             wrapInConstant1D(graph, index + 1),
                             wrapInConstant1D(graph, dim)});
}

torch::jit::Node *sliceHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // aten::slice(Tensor self, int dim, int start, int end, int step) -> Tensor
  // // NOLINT

  std::int64_t dim = constantToLong(node->input(1)->node());

  std::int64_t start = constantToLong(node->input(2)->node());

  std::int64_t end = constantToLong(node->input(3)->node());

  c10::TensorTypePtr as_tensor =
      node->input(0)->type()->cast<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  // Based on aten/src/ATen/native/TensorShape.cpp slice()
  if (start < 0) {
    start += *dims[dim];
  }
  if (end < 0) {
    end += *dims[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= *dims[dim]) {
    start = *dims[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= *dims[dim]) {
    end = *dims[dim];
  }

  return createSlice(graph, {node->input(0), wrapInConstant1D(graph, start),
                             wrapInConstant1D(graph, end),
                             wrapInConstant1D(graph, dim)});
}

torch::jit::Node *contiguousHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  // aten::contiguous(Tensor self, *, MemoryFormat
  // memory_format=contiguous_format) -> Tensor Returns a copy of the tensor but
  // in contiguous memory.
  //
  // aten::detach(Tensor self) -> Tensor
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

torch::jit::Node *splitChunkHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  // aten::split(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]"
  // aten::split(Tensor self, int split_sizes, int dim=0) -> Tensor[]"
  // aten::chunk(Tensor self, int chunks, int dim) -> Tensor[]

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

  if (kind == c10::aten::chunk) {
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
    // Create a slice.
    torch::jit::Node *slice =
        createSlice(graph, {node->input(0), wrapInConstant1D(graph, index),
                            wrapInConstant1D(graph, index + slice_size),
                            wrapInConstant1D(graph, axis)});

    // Add the slice to the graph.
    slices.push_back(slice->output());

    // Move along in the vector dimension.
    index += slice_size;
  }

  return createAndInsertNode(graph, at::prim::ListConstruct, slices);
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
      node->input(1)->type()->cast<c10::IntType>()) {
    cast_to = getNodeScalarType(node->output(0));
  }

  if (cast_to.has_value()) {
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
  // aten::upsample_bilinear2d(Tensor self, int[] output_size, bool
  // align_corners, float? scales_h, float? scales_w) -> Tensor
  //
  // aten::upsample_trilinear3d(Tensor self, int[] output_size, bool
  // align_corners, float? scales_d, float? scales_h, float? scales_w) -> Tensor

  std::vector<std::int64_t> output_size = shapeFromTensor(node->output());
  std::vector<std::int64_t> input_size = shapeFromTensor(node->input(0));
  std::vector<double> scales;
  ERROR_ON_MSG(output_size.size() != input_size.size(),
               "Input / output rank mismatch: " << input_size.size() << " != "
                                                << output_size.size());
  for (std::uint64_t d = 0; d < input_size.size(); ++d) {
    scales.push_back(static_cast<double>(output_size[d]) / input_size[d]);
  }

  torch::jit::Node *scales_node = createConstantFloat(
      graph, scales, {static_cast<std::int64_t>(scales.size())});
  return createResize(graph, {node->input(0), scales_node->output()},
                      "nearest");
}

torch::jit::Node *unsupportedUpsampleHandler(torch::jit::Graph *graph,
                                             torch::jit::Node *node) {
  UNUSED(graph);
  ERROR("Unsupported upsample mode "
        << node->kind().toQualString()
        << ": currently only 'nearest' is supported");
  return nullptr;
}

} // namespace

// clang-format off
static bool handlers = registerHandlers(
    c10::aten::expand, expandHandler,
    c10::aten::expand_as, expandAsHandler,
    c10::aten::view, reshapeHandler,
    c10::aten::unsqueeze, reshapeHandler,
    c10::aten::flatten, reshapeHandler,
    c10::aten::reshape, reshapeHandler,
    c10::aten::select,  selectHandler,
    c10::aten::slice,  sliceHandler,
    c10::aten::split,  splitChunkHandler,
    c10::aten::chunk,  splitChunkHandler,
    c10::aten::contiguous,  contiguousHandler,
    c10::aten::detach,  contiguousHandler,
    c10::aten::permute,  permuteHandler,
    c10::aten::transpose, transposeHandler,
    c10::aten::to, toHandler,
    c10::aten::upsample_nearest1d, upsampleHandler,
    c10::aten::upsample_nearest2d, upsampleHandler,
    c10::aten::upsample_nearest3d, upsampleHandler,
    c10::aten::upsample_linear1d, unsupportedUpsampleHandler,
    c10::aten::upsample_bilinear2d, unsupportedUpsampleHandler,
    c10::aten::upsample_trilinear3d, unsupportedUpsampleHandler,
    c10::aten::upsample_bicubic2d, unsupportedUpsampleHandler,
    c10::aten::squeeze, reshapeHandler);
// clang-format on

} // namespace poptorch
