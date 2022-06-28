// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/DispatchTracer.hpp"
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

torch::jit::Node *asStridedHandler(torch::jit::Graph * /*graph*/,
                                   torch::jit::Node * /*node*/) {
  // as_strided(Tensor(a) self, int[] size, int[] stride, int?
  // storage_offset=None) -> Tensor(a)

  // as_strided is very generic and as a result complex and expensive to handle.
  // However it is always generated as part of a decomposition so we should
  // catch whichever op is getting decomposed rather than deal with as_strided.
  ERROR(
      "InternalError: aten::as_strided should have been intercepted earlier.");
  return nullptr;
}

torch::jit::Node *reshapeHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
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
  auto *input = node->input(0);
  std::int64_t dim = constantToLong(node->input(1)->node());
  auto dims = shapeFromTensor(input);
  if (dim < 0) {
    dim += dims.size();
  }

  auto *index_node = node->input(2)->node();

  torch::jit::Node *slice_node;
  if (!isTensorConstant(index_node)) {
    // Handle dynamic index
    slice_node =
        createDynamicslice(graph, {input, index_node->output()}, {dim}, {1}, 1);
  } else {
    // Handle static index
    std::int64_t index = constantToLong(index_node);

    if (index < 0) {
      index += dims.at(dim);
    }

    slice_node = createSlice(graph, {input}, {index + 1}, {index}, {dim});
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
  replaceAllUsesWith(node->output(), node->input(0));
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

// Get the indices for im2col
std::vector<int64_t> getGatherIndices(int64_t orig_rows, int64_t orig_cols,
                                      int64_t kernel_size_x,
                                      int64_t kernel_size_y, int64_t dilation_x,
                                      int64_t dilation_y, int64_t padding_x,
                                      int64_t padding_y, int64_t extra_padding,
                                      int64_t stride_x, int64_t stride_y) {
  auto spatial_rows =
      (orig_rows + 2 * padding_y - dilation_y * (kernel_size_y - 1) - 1) /
          stride_y +
      1;
  auto spatial_cols =
      (orig_cols + 2 * padding_x - dilation_x * (kernel_size_x - 1) - 1) /
          stride_x +
      1;

  auto spatial_row_cols_product = spatial_rows * spatial_cols;

  auto numel = spatial_row_cols_product * kernel_size_x * kernel_size_y;

  std::vector<int64_t> indices;
  indices.reserve(numel);

  for (int64_t idx = 0; idx < numel; idx++) {
    auto kernel_offset = idx / spatial_row_cols_product;
    auto kernel_x_offset = (kernel_offset % kernel_size_x) * dilation_x;
    auto kernel_y_offset = (kernel_offset / kernel_size_x) * dilation_y;

    auto spatial_offset = idx % spatial_row_cols_product;
    auto spatial_x_offset = (spatial_offset % spatial_cols) * stride_x;
    auto spatial_y_offset = (spatial_offset / spatial_cols) * stride_y;

    auto actual_x = spatial_x_offset + kernel_x_offset;
    auto actual_y = spatial_y_offset + kernel_y_offset;

    auto in_idx =
        actual_y * (orig_cols + 2 * padding_x + extra_padding) + actual_x;

    if (actual_x < 0 || actual_y < 0) {
      ERROR("Out of range too low");
    }

    if (actual_x < 0 || actual_y < 0 ||
        actual_x >= (orig_cols + 2 * padding_x + 10) ||
        actual_y >= (orig_rows + 2 * padding_y)) {
      ERROR("Out of range");
    }
    indices.push_back(in_idx);
  }
  return indices;
}

// Reorder the padded im2col input to permit longer slices.
// Update supplied indices in place to match: these will have longer
// consecutive sequences.
torch::jit::Node *reorderBasedOnStride(torch::jit::Graph *graph,
                                       torch::jit::Value *padded,
                                       const std::vector<int64_t> &data_shape,
                                       int64_t stride, int64_t last_dim_size,
                                       std::vector<int64_t> *indices) {
  // Reshape to allow slicing based on index modulo stride
  auto *reshaped =
      createReshape(graph, padded, {data_shape[0], data_shape[1], -1, stride});

  // Slice and concatenate to order based on module stride
  std::vector<torch::jit::Value *> stride_sliced_flattened;
  stride_sliced_flattened.reserve(stride);

  for (int64_t start = 0; start < stride; start++) {
    auto *stride_sliced =
        createSlice(graph, {reshaped->output()}, {start + 1}, {start}, {3});
    auto *stride_flattened = createReshape(graph, stride_sliced->output(),
                                           {data_shape[0], data_shape[1], -1});
    stride_sliced_flattened.push_back(stride_flattened->output());
  }

  auto *concat = createConcat(graph, stride_sliced_flattened, 2);

  // Alter the indices to match
  for (size_t idx = 0; idx < indices->size(); idx++) {
    uint64_t old_idx = (*indices)[idx];
    (*indices)[idx] =
        (old_idx % stride) * (last_dim_size / stride) + old_idx / stride;
  }

  return concat;
}

// Convert indices to slices by accumulating consecutive indices into a single
// slice. Returns slice values as a pair (start, end).
std::vector<std::pair<int64_t, int64_t>>
indicesToSlices(const std::vector<int64_t> &indices) {
  ERROR_ON(indices.empty());

  // Represents the start and end of each slice in a pair
  std::vector<std::pair<int64_t, int64_t>> slices;

  int64_t slice_start = indices[0];
  for (auto it = indices.begin() + 1; it != indices.end(); it++) {
    auto previous = *(it - 1);
    auto current = *it;

    if (current != previous + 1) {
      slices.emplace_back(slice_start, previous + 1);
      slice_start = current;
    }
  }

  // Handle the last slice
  slices.emplace_back(slice_start, indices.back() + 1);

  return slices;
}

torch::jit::Node *im2colHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation,
  //              int[2] padding, int[2] stride) -> Tensor

  torch::jit::Value *data = node->input(0);
  std::vector<int64_t> data_shape = shapeFromTensor(data);
  ERROR_ON(data_shape.size() != 4);

  std::vector<std::int64_t> kernel_shape =
      constantToLongVec(node->input(1)->node());
  ERROR_ON(kernel_shape.size() != 2);

  std::vector<std::int64_t> dilation =
      constantToLongVec(node->input(2)->node());
  ERROR_ON(dilation.size() != 2);

  std::vector<std::int64_t> padding = constantToLongVec(node->input(3)->node());
  ERROR_ON(padding.size() != 2);

  std::vector<std::int64_t> strides = constantToLongVec(node->input(4)->node());
  ERROR_ON(strides.size() != 2);

  // First zero-pad the input
  // Pytorch gives the padding as being the amount to pad in both
  // directions. Popart has two arguments for each axis, the amount to pad in
  // each direction along that axis. In the form (Axis0Left, AxisNLeft...,
  // Axis0Right, AxisNRight) where left and right refer to the direction
  // along the axis to add zeros to.
  std::vector<std::int64_t> popart_padding{0, 0, padding[0], padding[1],
                                           0, 0, padding[0], padding[1]};

  // Increase RHS padding to ensure that the number of cols divides by the
  // x stride value
  auto current_width = data_shape[3] + padding[1] * 2;
  auto extra_padding = strides[1] - (current_width % strides[1]);
  extra_padding = extra_padding % strides[1];
  popart_padding.back() += extra_padding;
  current_width += extra_padding;

  auto *padded =
      createConstantPad(graph, node->input(0), popart_padding, 0., true);

  // Get the indices as if the spatial dimensions had been flattened
  auto indices =
      getGatherIndices(data_shape[2], data_shape[3], kernel_shape[1],
                       kernel_shape[0], dilation[1], dilation[0], padding[1],
                       padding[0], extra_padding, strides[1], strides[0]);

  // Calculate the last dim size as if it was flattened
  auto last_dim_size = current_width * (data_shape[2] + padding[0] * 2);

  // Reorder to allow fewer slices then each index became a slice
  auto *rearranged = reorderBasedOnStride(graph, padded->output(), data_shape,
                                          strides[1], last_dim_size, &indices);
  auto slices_start_end = indicesToSlices(indices);

  // Slice and concat for the reordering
  std::vector<torch::jit::Value *> sliced;
  sliced.reserve(slices_start_end.size());
  for (auto slice_start_end : slices_start_end) {
    sliced.push_back(createSlice(graph, {rearranged->output()},
                                 {slice_start_end.second},
                                 {slice_start_end.first}, {2})
                         ->output());
  }

  auto *concat = createConcat(graph, sliced, 2);

  // Finally reshape to match PyTorch's expectation
  return createReshape(
      graph, concat->output(),
      {data_shape[0], data_shape[1] * kernel_shape[0] * kernel_shape[1], -1});
}
// Make the scatter reduces indices for col2im
at::Tensor getScatterReduceIndices(int64_t num_cols, int64_t orig_rows,
                                   int64_t orig_cols, int64_t kernel_size_x,
                                   int64_t kernel_size_y, int64_t dilation_x,
                                   int64_t dilation_y, int64_t padding_x,
                                   int64_t padding_y, int64_t stride_x,
                                   int64_t stride_y) {
  // Add unity dimensions for batch and channel to facilitate tiling later
  auto indices = at::empty({1, 1, num_cols},
                           at::dtype(at::ScalarType::Int)
                               .memory_format(c10::MemoryFormat::Contiguous));

  auto *indices_ptr = indices.data_ptr<std::int32_t>();

  // The last dim has a mix of all kernel and spatial positions. Calculate
  // the number of spatial columns.
  auto spatial_cols =
      ((orig_cols + 2 * padding_x - dilation_x * (kernel_size_x - 1) - 1) /
       stride_x) +
      1;

  // spatial_rows*spatial_cols
  // (a short cut compared to calculating spatial_rows using the equivalent
  // expression used for spatial_cols)
  auto spatial_row_cols_product = num_cols / (kernel_size_x * kernel_size_y);

  // Find the original co-ordinate (x, y) from which the value in col_idx was
  // copied and calculate what the index would be
  for (int64_t col_idx = 0; col_idx < num_cols; col_idx++) {
    auto kernel_offset = col_idx / spatial_row_cols_product;
    auto kernel_x_offset = (kernel_offset % kernel_size_x) * dilation_x;
    auto kernel_y_offset = (kernel_offset / kernel_size_x) * dilation_y;

    auto spatial_offset = col_idx % (spatial_row_cols_product);
    auto spatial_x_offset = (spatial_offset % spatial_cols) * stride_x;
    auto spatial_y_offset = (spatial_offset / spatial_cols) * stride_y;

    auto actual_x = spatial_x_offset + kernel_x_offset - padding_x;
    auto actual_y = spatial_y_offset + kernel_y_offset - padding_y;

    auto index = actual_y * orig_cols + actual_x;

    // If out of range, use an out of range index. Poplar will skip this
    // index.
    if (actual_x < 0 || actual_y < 0 || actual_x >= orig_cols ||
        actual_y >= orig_rows) {
      index = orig_rows * orig_cols;
    }
    *indices_ptr = static_cast<int32_t>(index);
    indices_ptr++; // NOLINT
  }

  return indices;
}

torch::jit::Node *col2imHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::col2im(Tensor self, int[2] output_size, int[2] kernel_size,
  //              int[2] dilation, int[2] padding, int[2] stride) -> Tensor

  // This is somewhat of an inverse to im2col:
  // col2im(im2col(input)) == divisor * input with divisor as a tensor
  // im2col and col2im were used to speed up convolutions via GEMM.

  torch::jit::Value *data = node->input(0);
  std::vector<int64_t> data_shape = shapeFromTensor(data);
  ERROR_ON(data_shape.size() != 3);

  std::vector<std::int64_t> output_size =
      constantToLongVec(node->input(1)->node());
  ERROR_ON(output_size.size() != 2);

  std::vector<std::int64_t> kernel_shape =
      constantToLongVec(node->input(2)->node());
  ERROR_ON(kernel_shape.size() != 2);

  std::vector<std::int64_t> dilation =
      constantToLongVec(node->input(3)->node());
  ERROR_ON(dilation.size() != 2);

  std::vector<std::int64_t> padding = constantToLongVec(node->input(4)->node());
  ERROR_ON(padding.size() != 2);

  std::vector<std::int64_t> stride = constantToLongVec(node->input(5)->node());
  ERROR_ON(stride.size() != 2);

  // The batch and original channel ordering is unaffected by im2col so we can
  // reshape to factor them out.
  auto out_channels = data_shape[1] / (kernel_shape[0] * kernel_shape[1]);
  auto num_cols = data_shape[2] * (kernel_shape[0] * kernel_shape[1]);
  auto *reshaped =
      createReshape(graph, data, {data_shape[0], out_channels, num_cols});

  // Use scatter reduce to add across the relevent positions
  auto indices = getScatterReduceIndices(
      num_cols, output_size[0], output_size[1], kernel_shape[1],
      kernel_shape[0], dilation[1], dilation[0], padding[1], padding[0],
      stride[1], stride[0]);
  auto *indices_const = tensorToConstant(graph, indices);

  // The indices are shape (1, 1, num_cols) but need to be tiled for the
  // scatterreduce
  auto repeats =
      at::ones({3}, at::dtype(at::ScalarType::Long)
                        .memory_format(c10::MemoryFormat::Contiguous));
  repeats[0] = data_shape[0];
  repeats[1] = out_channels;
  auto *repeats_const = tensorToConstant(graph, repeats);
  auto *indices_tiled =
      createTile(graph, {indices_const->output(), repeats_const->output()});

  auto *scatter_reduced =
      createScatterreduce(graph, {reshaped->output(), indices_tiled->output()},
                          output_size[0] * output_size[1], 2, 0);

  return createReshape(
      graph, scatter_reduced->output(),
      {data_shape[0], out_channels, output_size[0], output_size[1]});
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
  c10::optional<size_t> size = dims.size();
  ERROR_ON_MSG(!size, std::string("Number of dimensions for tensor %") +
                          node->input(0)->debugName() + " is undefined. " +
                          "About to read uninitialized memory," +
                          " unexpected behaviour happened before transpose.");
  for (std::uint64_t i = 0; i < *size; ++i) {
    permutation.push_back(i);
  }

  // Allow for python array style access.
  if (dim0 < 0) {
    dim0 = *size + dim0;
  }

  if (dim1 < 0) {
    dim1 = *size + dim1;
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
    auto chunk_dim = *dims[axis];
    auto n_chunks = *split_size;

    // Integer division: (dim / n_chunks) with rounding up
    std::int64_t slice_size = (chunk_dim + n_chunks - 1) / n_chunks;
    auto remaining_size = chunk_dim;
    while (remaining_size >= slice_size) {
      size_of_each_split.push_back(slice_size);
      remaining_size -= slice_size;
    }
    // If we can't divide into equal chunks, then divide such that all but
    // the last chunk are the same size, and the last chunk is smaller.
    // If such a division is not possible, then return one fewer
    // chunks than specified
    if (remaining_size > 0) {
      // Add an extra slice for the remainder.
      size_of_each_split.push_back(remaining_size);
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
    torch::jit::Node *slice = createSlice(
        graph, {node->input(0)}, {index + slice_size}, {index}, {axis});

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
  std::vector<int64_t> v;
  for (std::uint64_t i = 0; i < *dims.size(); ++i) {
    v.push_back(*dims[i]);
  }
  // Propagate types
  for (size_t i = 0; i < slices.size(); i++) {
    v[axis] = size_of_each_split[i];
    auto type = slices[i]->type()->expect<c10::TensorType>()->withSizes(v);
    unpack->output(i)->setType(type);
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
    if (*cast_to == at::ScalarType::Float && !isCompilingWithDispatcher()) {
      cast_to = HALF_OR_FLOAT;
    }

    // Avoid promoting to an unsupported type
    cast_to = coerceToSupportedType(*cast_to);
  }

  if (!cast_to.has_value() || cast_to == *tensor_type->scalarType()) {
    // NOOP
    if (cast_to == *tensor_type->scalarType()) {
      logging::trace("Ignoring type cast to same type, {}, {}", *cast_to,
                     *tensor_type->scalarType());
    }

    replaceAllUsesWith(node->output(), node->input(0));
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

torch::jit::Node *upsampleBicubic2dHandler(torch::jit::Graph *graph,
                                           torch::jit::Node *node) {
  // upsample_bicubic2d(Tensor self, int[2] output_size, bool align_corners,
  // float? scales_h=None, float? scales_w=None) -> Tensor

  torch::jit::Value *input = node->input(0);
  torch::jit::Value *output_size = node->input(1);
  torch::jit::Value *output_scale = node->input(3);

  auto align_corners = constantToBool(node->input(2)->node());
  ERROR_ON_MSG(align_corners, "Only support align_corners=False.");

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
  return createResize(graph, {input, scales_node->output()}, "cubic");
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
    replaceAllUsesWith(node->output(), node->input(0));
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
  registerHandler(c10::aten::col2im, col2imHandler);
  registerHandler(c10::aten::im2col, im2colHandler);
  registerHandler(c10::aten::numpy_T, numpyTHandler);
  registerHandler(c10::aten::to, toHandler);
  registerHandler(c10::aten::type_as, toHandler);
  registerHandler(c10::aten::upsample_nearest1d, upsampleHandler);
  registerHandler(c10::aten::upsample_nearest2d, upsampleHandler);
  registerHandler(c10::aten::upsample_nearest3d, upsampleHandler);
  registerHandler(c10::aten::upsample_linear1d, unsupportedUpsampleHandler);
  registerHandler(c10::aten::upsample_bilinear2d, upsampleBilinear2dHandler);
  registerHandler(c10::aten::upsample_trilinear3d, unsupportedUpsampleHandler);
  registerHandler(c10::aten::upsample_bicubic2d, upsampleBicubic2dHandler);
  registerHandler(c10::aten::squeeze, reshapeHandler);
  registerHandler(c10::aten::as_strided, asStridedHandler);
  registerHandler(c10::aten::stack, stackHandler);
  registerHandler(c10::aten::Int, intHandler);
  registerHandler(symbols::poptorch::autocast, autocastHandler);
}

} // namespace poptorch
