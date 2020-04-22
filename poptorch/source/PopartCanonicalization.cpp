#include <poptorch/OpBuilder.hpp>
#include <poptorch/PopartCanonicalization.hpp>

#include <torch/csrc/jit/ir/ir.h>
#include <unordered_set>

namespace poptorch {

namespace {

class CanonicalizeImpl {
public:
  void Run(torch::jit::Graph &graph);

private:
  // When we transform a node mark it for deletion, this will also clean up
  // unused users afterwards.
  std::unordered_set<torch::jit::Node *> toDelete;

  // Todo, template this later.
  std::vector<std::int64_t> HandleListConstruct(torch::jit::Node *node);

  template <typename T> std::optional<T> HandleConstant(torch::jit::Node *node);

  // Delete a node and also its users if they are also unused.
  void SearchAndPossiblyDestroy(torch::jit::Node *node);

  // Pytorch IR represents 'None' as a constant with no value.
  bool IsNone(torch::jit::Node *node) const;
};

/*
 * Helper structs to help deduce the attribute types.
 */

template <typename T> struct Handle {};

template <> struct Handle<std::int64_t> {
  std::int64_t operator()(c10::Symbol &sym, torch::jit::Node *node) {
    return node->i(sym);
  }
};

template <> struct Handle<float> {
  float operator()(c10::Symbol &sym, torch::jit::Node *node) {
    return node->f(sym);
  }
};

/*
 * ConvertAtenToPopart implementation.
 */

// Some operations take in an optional tensor. A "none" constant is passed in to
// mark a tensor which is not there.
bool CanonicalizeImpl::IsNone(torch::jit::Node *node) const {
  if (std::string(node->kind().toDisplayString()) != "prim::Constant") {
    return false;
  }

  auto sym = c10::Symbol::fromQualString("attr::value");
  if (node->hasAttribute(sym)) {
    return false;
  }

  return true;
}

template <typename T>
std::optional<T> CanonicalizeImpl::HandleConstant(torch::jit::Node *node) {
  if (std::string(node->kind().toDisplayString()) != "prim::Constant") {
    std::cerr << "Constant is expected to be prim::Constant but is "
              << node->kind().toDisplayString() << std::endl;
    assert(false && "Constant kind must be prim::Constant");
    return std::nullopt;
  }

  auto sym = c10::Symbol::fromQualString("attr::value");

  if (!node->hasAttribute(sym)) {
    return std::nullopt;
  }

  return Handle<T>{}(sym, node);
}

std::vector<std::int64_t>
CanonicalizeImpl::HandleListConstruct(torch::jit::Node *node) {
  std::vector<std::int64_t> result;

  for (torch::jit::Value *value : node->inputs()) {

    std::optional<std::int64_t> val =
        HandleConstant<std::int64_t>(value->node());
    if (val) {
      result.push_back(*val);
    }
  }

  return result;
}

void CanonicalizeImpl::SearchAndPossiblyDestroy(torch::jit::Node *node) {

  // Skip parameters and nodes with any uses.
  if (node->kind() == c10::prim::Param || node->hasUses()) {
    return;
  }

  // Store the inputs used by this node.
  std::vector<torch::jit::Value *> inputs;
  for (torch::jit::Value *user : node->inputs()) {
    inputs.push_back(user);
  }

  // Delete the node.
  node->destroy();

  // If any of the previously used values now have no users repeat the process
  // for them.
  for (torch::jit::Value *user : inputs) {
    SearchAndPossiblyDestroy(user->node());
  }
}

void CanonicalizeImpl::Run(torch::jit::Graph &graph) {
  for (torch::jit::Node *node : graph.nodes()) {
    torch::jit::Node *newNode = nullptr;

    torch::jit::Symbol kind = node->kind();
    std::string kindAsStr = kind.toDisplayString();

    if (kindAsStr == "aten::_convolution") {
      /*
      aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[]
      stride, int[] padding, int[] dilation, bool transposed, int[]
      output_padding, int groups) -> Tensor
      */
      std::optional<std::int64_t> transposed =
          HandleConstant<std::int64_t>(node->inputs()[6]->node());

      torch::jit::Value *input = node->inputs()[0];
      torch::jit::Value *kernel = node->inputs()[1];

      std::vector<torch::jit::Value *> inputs{input, kernel};

      if (!IsNone(node->inputs()[2]->node())) {
        inputs.push_back(node->inputs()[2]);
      }

      std::vector<std::int64_t> stride =
          HandleListConstruct(node->inputs()[3]->node());
      std::vector<std::int64_t> padding =
          HandleListConstruct(node->inputs()[4]->node());

      // Slight workaround for current padding mechanism here.
      padding.push_back(padding[0]);
      padding.push_back(padding[1]);

      std::vector<std::int64_t> dilation =
          HandleListConstruct(node->inputs()[5]->node());
      // torch::jit::Value* output_padding = node->inputs()[8];
      std::int64_t groups =
          *HandleConstant<std::int64_t>(node->inputs()[8]->node());

      if (transposed && *transposed == 0) {

        // Create a "normal" convolution.
        newNode = poptorch::Create_conv(graph, inputs, dilation, groups, {},
                                        padding, stride);

      } else {

        std::cerr << "CURRENTLY UNSUPPORTED CONVOLUTION!!!";
        newNode->dump();
      }
    } else if (kindAsStr == "aten::batch_norm") {
      /*
      aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor?
      running_mean, Tensor?  , bool training, float momentum, float
      eps, bool cudnn_enabled) -> Tensor
      */

      torch::jit::Value *input = node->inputs()[0];
      torch::jit::Value *weight = node->inputs()[1];
      torch::jit::Value *bias = node->inputs()[2];
      torch::jit::Value *running_mean = node->inputs()[3];
      torch::jit::Value *running_var = node->inputs()[4];

      // TODO: These will have to be checked if they are actual tensors in the
      // future.
      std::vector<torch::jit::Value *> inputTensors{input, weight, bias,
                                                    running_mean, running_var};

      std::int64_t training =
          0; //*HandleConstant<std::int64_t>(node->inputs()[5]->node());
      float momentum = *HandleConstant<float>(node->inputs()[6]->node());
      float epsilon = *HandleConstant<float>(node->inputs()[7]->node());

      newNode = poptorch::Create_batchnormalization(graph, inputTensors, 1,
                                                    epsilon, momentum);

    } else if (kindAsStr == "aten::max_pool2d") {
      /*
        aten::max_pool2d(Tensor self, int[] kernel_size, int[] stride, int[]
        padding, int[] dilation, bool ceil_mode) -> Tensor
     */
      std::vector<std::int64_t> kernel_size =
          HandleListConstruct(node->inputs()[1]->node());
      std::vector<std::int64_t> stride =
          HandleListConstruct(node->inputs()[2]->node());
      std::vector<std::int64_t> padding =
          HandleListConstruct(node->inputs()[3]->node());
      std::vector<std::int64_t> dilation =
          HandleListConstruct(node->inputs()[4]->node());

      // Slight workaround for current padding mechanism here.
      padding.push_back(padding[0]);
      padding.push_back(padding[1]);

      newNode = poptorch::Create_maxpool(graph, {node->inputs()[0]}, 1,
                                         kernel_size, padding, 0, stride);
    } else if (kindAsStr == "aten::add") {
      // Drop the nonsense term in the add.
      // TODO: Figure out what the "alpha" is.
      newNode =
          poptorch::Create_add(graph, {node->inputs()[0], node->inputs()[1]});
    } else if (kindAsStr == "aten::flatten") {

      newNode = poptorch::Create_flatten(graph, {node->inputs()[0]}, 1);
    } else if (kindAsStr == "aten::addmm") {

      //"aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta,
      // Scalar alpha) -> Tensor")
      torch::jit::Value *matA = node->inputs()[0];
      torch::jit::Value *matB = node->inputs()[1];
      torch::jit::Value *matC = node->inputs()[2];

      float beta = *HandleConstant<std::int64_t>(node->inputs()[3]->node());
      float alpha = *HandleConstant<std::int64_t>(node->inputs()[4]->node());

      newNode = poptorch::Create_matmul(graph, {matB, matC});
      newNode->insertBefore(node);
      newNode = poptorch::Create_add(graph, {newNode->output(), matA});
      //  newNode = poptorch::CreateAdd(graph, matA, matB);
      //  newNode = poptorch::CreateGEMM(graph, matB, matC, matA, beta, alpha);

    } else if (kindAsStr == "aten::adaptive_avg_pool2d") {

      std::vector<std::int64_t> outputShape =
          HandleListConstruct(node->inputs()[1]->node());

      c10::TensorTypePtr asTensor =
          node->inputs()[0]->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();
      std::vector<std::int64_t> inputShape{*dims[2], *dims[3]};

      // Need to clean this code up.
      // TODO.
      const std::vector<int64_t> &stride{inputShape[0] / outputShape[0],
                                         inputShape[1] / outputShape[1]};

      const std::vector<int64_t> &kernel_shape{
          inputShape[0] - (outputShape[0] - 1) * stride[0],
          inputShape[1] - (outputShape[1] - 1) * stride[1]};
      const std::vector<int64_t> &padding{0, 0, 0, 0};

      newNode = Create_averagepool(graph, {node->inputs()[0]}, kernel_shape, 0,
                                   padding, stride);
    } else if (kindAsStr == "aten::softmax") {
      // "aten::softmax(Tensor self, int dim, int? dtype) -> Tensor"

      std::int64_t dim =
          *HandleConstant<std::int64_t>(node->inputs()[1]->node());

      newNode = Create_softmax(graph, {node->inputs()[0]}, dim);

    } else if (kindAsStr == "aten::view") {
      // aten::view(Tensor(a) self, int[] size) -> Tensor(a)

      /*
      // This might be needed if we can't rely on the return type being there.
      std::vector<std::int64_t> newShape =
          HandleListConstruct(node->inputs()[1]->node());*/

      // Obviously we can't support a varying shape so we might need to change
      // this.

      c10::TensorTypePtr asTensor =
          node->output()->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();

      std::vector<std::int64_t> newShape{*dims[0], *dims[1]};

      newNode = CreateReshape(graph, node->inputs()[0], newShape);
    } else if (kindAsStr == "aten::dropout") {
      // aten::dropout(Tensor input, float p, bool train) -> Tensor

      float rate = *HandleConstant<float>(node->inputs()[1]->node());

      newNode = Create_dropout(graph, {node->inputs()[0]}, 1, rate);
    } else if (kindAsStr == "poptorch::begin_ipu_block") {

      // This could maybe be improved. Can we add attributes on the frontend?
      // TODO.
      newNode = graph.create(
          c10::Symbol::fromQualString("poptorch::begin_ipu_block"));

      // Convert the prim::Constant into an attribute.
      std::int64_t ipu_id =
          *HandleConstant<std::int64_t>(node->input()->node());
      newNode->i_(c10::Symbol::fromQualString("attr::ipu"), ipu_id);
    } else if (kindAsStr == "aten::t") {
      newNode = Create_transpose(graph, {node->inputs()[0]}, {});
    } else if (kindAsStr == "aten::relu" || kindAsStr == "aten::relu_") {
      newNode = Create_relu(graph, {node->inputs()[0]});
    }

    // If we have a new node add it and replace the old use.
    if (newNode) {
      newNode->insertBefore(node);

      // Mark this node for deletion.
      toDelete.insert(node);

      if (node->hasUses()) {
        torch::jit::Value *newVal = newNode->output();
        torch::jit::Value *oldVal = node->output();

        // Take the type of the old value.
        newVal->setType(oldVal->type());

        // Replace the old value with the new one.
        oldVal->replaceAllUsesWith(newVal);
      }
    }
  }

  // Remove any dead nodes.
  for (torch::jit::Node *node : toDelete) {
    SearchAndPossiblyDestroy(node);
  }
}

} // namespace

void Canonicalize(torch::jit::Graph &graph) {
  CanonicalizeImpl converter;
  converter.Run(graph);
}

} // namespace poptorch
