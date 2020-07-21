// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <utility>

#include "poptorch/ShapeInference.hpp"
#include "poptorch_logging/Logging.hpp"

#include "PoptorchSymbols.hpp"

namespace poptorch {

using InferenceFunction = std::function<void(torch::jit::Node *)>;

class InferenceFunctions {
public:
  InferenceFunctions(const InferenceFunctions &) = delete;
  InferenceFunctions &operator=(const InferenceFunctions &) = delete;

  static void registerFunction(const std::vector<c10::Symbol> &kinds,
                               const InferenceFunction &func) {
    for (auto &kind : kinds) {
      registerFunction(kind, func);
    }
  }

  static void registerFunction(c10::Symbol kind,
                               const InferenceFunction &func) {
    instance()._inference_functions.insert({kind, func});
  }

  static void tryInferShapeFor(torch::jit::Node *node) {
    c10::Symbol kind = node->kind();
    auto found = instance()._inference_functions.find(kind);
    if (found != instance()._inference_functions.end()) {
      auto inference_func = found->second;
      inference_func(node);
    } else {
      std::cerr << "Warning: Dont know how to infer shape for node of kind '"
                << kind.toDisplayString() << "'\n";
    }
  }

private:
  InferenceFunctions() = default;

  // `instance` is not static member of the class as the initialization of
  // globals is undefined. Accessing `instance` through a method forces it to be
  // constructed before use.
  // https://stackoverflow.com/questions/3746238/c-global-initialization-order-ignores-dependencies/3746249#3746249
  static InferenceFunctions &instance() {
    static InferenceFunctions instance;
    return instance;
  }

  std::unordered_map<c10::Symbol, InferenceFunction> _inference_functions;
};

struct RegisterInferenceFunction {
  RegisterInferenceFunction(const std::vector<c10::Symbol> &kinds,
                            const InferenceFunction &func) {
    InferenceFunctions::registerFunction(kinds, func);
  }

  RegisterInferenceFunction(c10::Symbol kind, const InferenceFunction &func) {
    InferenceFunctions::registerFunction(kind, func);
  }
};

void outputTypeMatchesInputType(torch::jit::Node *node) {
  auto input_type = node->input(0)->type()->cast<torch::jit::TensorType>();
  if (!input_type->isComplete()) {
    logging::err("Cannot infer shape, input shape is not complete!");
    return;
  }

  node->output()->setType(input_type);
}

bool isInputACompleteTensor(torch::jit::Node *node, int index) {
  if (!node->input(index)->type()->isSubtypeOf(torch::jit::TensorType::get())) {
    return false;
  }
  auto i = node->input(index)->type()->cast<torch::jit::TensorType>();
  return i->isComplete();
}

void inferShapeFlatten(torch::jit::Node *node) {
  if (!isInputACompleteTensor(node, 0)) {
    logging::err("Cannot infer shape, first input is either not a tensor or "
                 "has an incomplete shape.");
    return;
  }
  auto i0 = node->input(0)->type()->cast<torch::jit::TensorType>();
  auto i0_shape = *i0->sizes().concrete_sizes();

  if (node->input(1)->node()->kind() != c10::prim::Constant) {
    logging::err("Cannot infer shape, unable to get data for shape input ");
    return;
  }
  auto start = node->input(1)->node()->i(c10::attr::value);

  if (node->input(1)->node()->kind() != c10::prim::Constant) {
    logging::err("Cannot infer shape, unable to get data for shape input.");
    return;
  }
  auto end = node->input(2)->node()->i(c10::attr::value);

  if (end == -1) {
    end = i0_shape.size() - 1;
  }

  std::vector<int64_t> result_shape;
  result_shape.reserve(start);
  for (int i = 0; i < start; i++) {
    result_shape.push_back(i0_shape.at(i));
  }

  int64_t x = 1;
  for (int i = start; i <= end; i++) {
    x *= i0_shape.at(i);
  }
  result_shape.push_back(x);

  for (uint i = end + 1; i < i0_shape.size(); i++) {
    result_shape.push_back(i0_shape.at(i));
  }

  auto output_type = i0->withSizes(result_shape);
  node->output()->setType(output_type);
}

void inferShapeAdaptiveAvgPool2d(torch::jit::Node *node) {
  if (!isInputACompleteTensor(node, 0)) {
    logging::err("Cannot infer shape, first input is either not a tensor or "
                 "has an incomplete shape.");
    return;
  }
  auto i0 = node->input(0)->type()->cast<torch::jit::TensorType>();
  auto i0_shape = *i0->sizes().concrete_sizes();

  if (node->input(1)->node()->kind() != c10::prim::Constant) {
    logging::err("Cannot infer shape, unable to get data for shape input.");
    return;
  }
  auto i1_data = node->input(1)->node()->is(c10::attr::value);

  std::vector<int64_t> result_shape{i0_shape.at(0)};
  if (i0_shape.size() == 4) {
    result_shape.push_back(i0_shape.at(1));
  }
  for (auto i : i1_data) {
    result_shape.push_back(i);
  }

  auto output_type = i0->withSizes(result_shape);
  node->output()->setType(output_type);
}

void inferShapeBroadcast(torch::jit::Node *node) {
  if (!node->input(0)->type()->isSubtypeOf(torch::jit::TensorType::get())) {
    logging::err("Cannot infer shape, first input is not a Tensor.");
    return;
  }
  auto i0 = node->input(0)->type()->cast<torch::jit::TensorType>();
  if (!i0->isComplete()) {
    logging::err("Cannot infer shape, first input shape is not complete!");
    return;
  }

  if (!node->input(1)->type()->isSubtypeOf(torch::jit::TensorType::get())) {
    logging::err("Cannot infer shape, first input is not a Tensor.");
    return;
  }
  auto i1 = node->input(1)->type()->cast<torch::jit::TensorType>();
  if (!i1->isComplete()) {
    logging::err("Cannot infer shape, second input shape is not complete!");
    return;
  }

  std::vector<int64_t> i0_shape = *i0->sizes().concrete_sizes();
  std::vector<int64_t> i1_shape = *i1->sizes().concrete_sizes();

  std::int64_t i0_end = i0_shape.size() - 1;
  std::int64_t i1_end = i1_shape.size() - 1;
  std::vector<int64_t> result_shape;
  while (i0_end >= 0 && i1_end >= 0) {
    auto a = i0_shape.at(i0_end);
    i0_end--;
    auto b = i1_shape.at(i1_end);
    i1_end--;

    if (a == b || b == 1) {
      result_shape.push_back(a);
    } else if (a == 1) {
      result_shape.push_back(b);
    } else {
      logging::err(
          "Cannot broadcast shapes {} and {}. Dimensions {} and {} conflict.",
          i0_shape, i1_shape, a, b);
      return;
    }
  }
  while (i0_end >= 0 || i1_end >= 0) {
    if (i0_end >= 0) {
      auto a = i0_shape.at(i0_end);
      result_shape.push_back(a);
    } else {
      auto b = i1_shape.at(i1_end);
      result_shape.push_back(b);
    }

    i0_end--;
    i1_end--;
  }

  std::reverse(result_shape.begin(), result_shape.end());
  auto output_type = i0->withSizes(result_shape);
  node->output()->setType(output_type);
}

// aten::conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[]
// padding, int[] dilation, int groups) -> Tensor",
void inferShapeConv2d(torch::jit::Node *node) {
  auto input_type = node->input(0)->type()->cast<torch::jit::TensorType>();
  if (!input_type->isComplete()) {
    logging::err("Cannot infer shape, input shape is not complete!");
    return;
  }

  auto weight_type = node->input(1)->type()->cast<torch::jit::TensorType>();
  if (!weight_type->isComplete()) {
    logging::err("Cannot infer shape, weight shape is not complete!");
    return;
  }

  int stride_input_index = 3;
  int padding_input_index = 4;
  int dilation_input_index = 5;

  if (node->input(stride_input_index)->node()->kind() != c10::prim::Constant) {
    logging::err("Cannot infer shape, unable to get stride");
    return;
  }
  auto stride_data =
      node->input(stride_input_index)->node()->is(c10::attr::value);

  if (node->input(padding_input_index)->node()->kind() != c10::prim::Constant) {
    logging::err("Cannot infer shape, unable to get padding\n");
    return;
  }
  auto padding_data =
      node->input(padding_input_index)->node()->is(c10::attr::value);

  if (node->input(dilation_input_index)->node()->kind() !=
      c10::prim::Constant) {
    logging::err("Cannot infer shape, unable to get dilation\n");
    return;
  }
  auto dilation_data =
      node->input(dilation_input_index)->node()->is(c10::attr::value);

  auto input_shape = *input_type->sizes().concrete_sizes();
  auto weight_shape = *weight_type->sizes().concrete_sizes();

  auto batch_size = input_shape[0];
  auto out_chans = weight_shape[0];
  std::vector<int64_t> kernel_shape;
  kernel_shape.reserve(2);
  for (int i = 0; i < 2; i++) {
    kernel_shape.push_back(weight_shape[2 + i]);
  }

  auto calculate_dim = [](int64_t i, int64_t k, int64_t s, int64_t p,
                          int64_t d) {
    auto x = i + 2 * p - d * (k - 1) - 1;
    return (x / s) + 1;
  };

  std::vector<int64_t> out_shape{batch_size, out_chans};

  for (int i = 0; i < 2; i++) {
    auto x = calculate_dim(input_shape[2 + i], kernel_shape[i], stride_data[i],
                           padding_data[i], dilation_data[i]);
    out_shape.push_back(x);
  }

  auto output_type = input_type->withSizes(out_shape);
  node->output()->setType(output_type);
}

// "aten::max_pool2d(Tensor self, int[] kernel_size, int[] stride, int[]
// padding, int[] dilation, bool ceil_mode) -> Tensor",
void inferShapeMaxPool2d(torch::jit::Node *node) {
  auto input_type = node->input(0)->type()->cast<torch::jit::TensorType>();
  if (!input_type->isComplete()) {
    logging::err("Cannot infer shape, input shape is not complete!\n");
    return;
  }

  auto input_shape = *input_type->sizes().concrete_sizes();
  auto n = input_shape[0];
  auto c = input_shape[1];

  int kernel_input_index = 1;
  int stride_input_index = 2;
  int padding_input_index = 3;
  int dilation_input_index = 4;

  if (node->input(kernel_input_index)->node()->kind() != c10::prim::Constant) {
    logging::err("Cannot infer shape, unable to get kernel\n");
    return;
  }
  auto kernel_data =
      node->input(kernel_input_index)->node()->is(c10::attr::value);

  if (node->input(stride_input_index)->node()->kind() != c10::prim::Constant) {
    logging::err("Cannot infer shape, unable to get stride\n");
    return;
  }
  auto stride_data =
      node->input(stride_input_index)->node()->is(c10::attr::value);

  if (node->input(padding_input_index)->node()->kind() != c10::prim::Constant) {
    logging::err("Cannot infer shape, unable to get padding\n");
    return;
  }
  auto padding_data =
      node->input(padding_input_index)->node()->is(c10::attr::value);

  if (node->input(dilation_input_index)->node()->kind() !=
      c10::prim::Constant) {
    logging::err("Cannot infer shape, unable to get dilation\n");
    return;
  }
  auto dilation_data =
      node->input(dilation_input_index)->node()->is(c10::attr::value);

  std::vector<int64_t> out_shape{n, c};

  for (int i = 0; i < 2; i++) {
    auto stride = stride_data[i];
    auto padding = padding_data[i];
    auto kernel = kernel_data[i];
    auto dilation = dilation_data[i];
    auto dim = input_shape[2 + i];
    auto x = dim + 2 * padding - dilation * (kernel - 1) - 1;
    auto y = (x / stride) + 1;
    out_shape.push_back(y);
  }

  auto output_type = input_type->withSizes(out_shape);
  node->output()->setType(output_type);
}

void inferShapeView(torch::jit::Node *node) {
  auto input_type = node->input(0)->type()->cast<torch::jit::TensorType>();
  if (!input_type->isComplete()) {
    logging::err("Cannot infer shape, input shape is not complete!\n");
    return;
  }

  auto input_shape = *input_type->sizes().concrete_sizes();

  if (node->input(1)->node()->kind() != c10::prim::Constant) {
    logging::err("Cannot infer shape, unable to get data for shape input\n");
    return;
  }
  auto shape_data = node->input(1)->node()->is(c10::attr::value);

  auto get_number_elements = [](auto shape) {
    int64_t number_elements = 1;
    for (auto dim : shape) {
      if (dim != -1) {
        number_elements *= dim;
      }
    }
    return number_elements;
  };

  int unknown_dimensions = 0;
  for (auto dim : shape_data) {
    if (dim == -1) {
      unknown_dimensions += 1;
    }
  }

  if (unknown_dimensions == 0) {
    if (get_number_elements(shape_data) != get_number_elements(input_shape)) {
      logging::err("Error: New shape has a different number of elements to old "
                   "shape.\n");
      return;
    }
    auto output_type = input_type->withSizes(shape_data);
    node->output()->setType(output_type);
  } else if (unknown_dimensions == 1) {
    auto old_number_elements = get_number_elements(input_shape);
    auto new_number_elements = get_number_elements(shape_data);
    std::vector<int64_t> output_shape;
    for (auto dim : shape_data) {
      if (dim == -1) {
        output_shape.push_back(old_number_elements / new_number_elements);
      } else {
        output_shape.push_back(dim);
      }
    }
    auto output_type = input_type->withSizes(output_shape);
    node->output()->setType(output_type);
  } else {
    logging::err("Too many unknown dimensions ({}) in shape data ({})",
                 unknown_dimensions, shape_data);
  }
}

void inferShapeAddmm(torch::jit::Node *node) {
  auto mat1_type = node->input(1)->type()->cast<torch::jit::TensorType>();
  if (!mat1_type->isComplete()) {
    logging::err("Cannot infer shape, mat1 shape is not complete!");
    return;
  }

  auto mat1_shape = *mat1_type->sizes().concrete_sizes();

  auto mat2_type = node->input(2)->type()->cast<torch::jit::TensorType>();
  if (!mat2_type->isComplete()) {
    logging::err("Cannot infer shape, mat2 shape is not complete!");
    return;
  }

  auto mat2_shape = *mat2_type->sizes().concrete_sizes();

  auto n = mat1_shape[0];
  auto p = mat2_shape[1];

  std::vector<int64_t> out_shape{n, p};
  auto output_type = mat1_type->withSizes(out_shape);
  node->output()->setType(output_type);
}

void inferShapeTranspose(torch::jit::Node *node) {
  auto input_type = node->input(0)->type()->cast<torch::jit::TensorType>();
  if (!input_type->isComplete()) {
    logging::err("Cannot infer shape, input shape is not complete!");
    return;
  }

  auto input_shape = *input_type->sizes().concrete_sizes();
  if (input_shape.size() != 2) {
    logging::err("Transpose inference only handles inputs with rank 2.\n");
  }

  std::vector<int64_t> out_shape{input_shape[1], input_shape[0]};
  auto output_type = input_type->withSizes(out_shape);
  node->output()->setType(output_type);
}

void propagateInputShapes(torch::jit::Block *block) {
  for (auto node : block->nodes()) {
    for (auto b : node->blocks()) {
      propagateInputShapes(b);
    }

    InferenceFunctions::tryInferShapeFor(node);
  }
}

void propagateInputShapes(torch::jit::Graph *graph) {
  propagateInputShapes(graph->block());
}

namespace {
RegisterInferenceFunction output_matches_input(
    {c10::aten::batch_norm, c10::aten::relu, c10::aten::relu_,
     c10::aten::softmax, c10::prim::unchecked_cast},
    outputTypeMatchesInputType);
RegisterInferenceFunction conv2d(c10::aten::conv2d, inferShapeConv2d);
RegisterInferenceFunction maxpool2d(c10::aten::max_pool2d, inferShapeMaxPool2d);
RegisterInferenceFunction view(c10::aten::view, inferShapeView);
RegisterInferenceFunction addmm(c10::aten::addmm, inferShapeAddmm);
RegisterInferenceFunction transpose(c10::aten::t, inferShapeTranspose);
RegisterInferenceFunction broadcast({c10::aten::add, c10::aten::add_,
                                     c10::aten::sub, c10::aten::sub_,
                                     c10::aten::mul, c10::aten::mul_},
                                    inferShapeBroadcast);
RegisterInferenceFunction avgpool2d(c10::aten::adaptive_avg_pool2d,
                                    inferShapeAdaptiveAvgPool2d);
RegisterInferenceFunction flatten(c10::aten::flatten, inferShapeFlatten);

// These nodes have a dummy function registered so they are ignored.
RegisterInferenceFunction
    ignore({c10::prim::Constant, c10::aten::__getitem__, c10::aten::__is__,
            c10::aten::__isnot__, c10::aten::eq, c10::aten::ne, c10::aten::dim,
            c10::aten::len, c10::aten::size},
           [](auto node) {
             logging::err("Warning: Shape inference is ignoring node:\n  {}",
                          *node);
           });
} // namespace

} // namespace poptorch
