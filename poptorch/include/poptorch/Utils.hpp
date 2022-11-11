// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_UTILS_HPP
#define INCLUDE_POPTORCH_UTILS_HPP

#include <torch/csrc/jit/ir/ir.h>

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>

#include <vector>

namespace poptorch {

torch::jit::Node *findEarliestUser(const torch::jit::Value *value);

bool isNondeterministic(const torch::jit::Node &node);

std::string nodeToString(const torch::jit::Node *node);

std::string scalarTypeToOnnxString(at::ScalarType type);

at::ScalarType onnxStrToScalarType(const char *type_str);

at::ScalarType coerceToSupportedType(at::ScalarType type);

// Returns a collapsed version of the graph input hierachy into a list of
// tensor values by following any tuples/lists and their unpacking
// N.B. if a tuple is not used (unpacked), the resulting values will be null
// as a placeholder.
std::vector<torch::jit::Value *>
collapsedGraphInputHierachy(torch::jit::Graph *graph);

// Return the number of tensors for a given type: in the case of a tensor
// this is 1, but in case of nested tuples, this is the sum over all.
size_t numTensorsForType(const c10::TypePtr &type);

// Delete a node and also its inputs if they are also unused.
void searchAndPossiblyDestroy(
    const std::unordered_set<torch::jit::Node *> &to_test);

// Remove all the node's inputs and destroy them if they're not used
// anywhere else.
void removeAndPossiblyDestroyAllInputs(torch::jit::Node *node);

// Use unused type BFLOAT16 to indicate ambiguity between FLOAT16 and FLOAT32
// NOLINTNEXTLINE
const auto HALF_OR_FLOAT = at::ScalarType::BFloat16;

std::unique_ptr<char[]> stringToUniquePtr(const std::string &str);

// Get the tensor shape as a vector of ints.
std::vector<std::int64_t> shapeFromTensor(torch::jit::Value *value);

// Add casts as necessary such that weight and bias have the same scalar type
// as input.
void castWeightAndBias(torch::jit::Graph *graph, torch::jit::Value *input,
                       torch::jit::Value *&weight, torch::jit::Value *&bias);

// A replacement for PyTorch's ListType which includes the number of elements
// unlike PyTorch's own type.
class ListTypeWithNumElements
    : public c10::SingleElementType<torch::jit::TypeKind::ListType,
                                    ListTypeWithNumElements> {
public:
  ListTypeWithNumElements(c10::TypePtr elem_type, size_t num_elements)
      : SingleElementType(std::move(elem_type)), _num_elements(num_elements) {}

  bool equals(const Type &rhs) const override {
    if (auto rhs_cast = rhs.cast<ListTypeWithNumElements>()) {
      return numElements() == rhs_cast->numElements();
    }
    return false;
  }

  size_t numElements() const { return _num_elements; }

  std::string str() const override {
    std::stringstream ss;
    ss << "TensorList[" << _num_elements << "]";
    return ss.str();
  }

  c10::ListTypePtr getOriginalListType() const {
    return c10::ListType::create(getElementType());
  }

private:
  size_t _num_elements;

  std::string annotation_str_impl(c10::TypePrinter printer) const override {
    (void)(printer);
    return str();
  }
};
using ListTypeWithNumElementsPtr = std::shared_ptr<ListTypeWithNumElements>;

struct JitTensorInfo {
  explicit JitTensorInfo(const at::Tensor &tensor);
  explicit JitTensorInfo(torch::jit::Value *value);
  std::string toString() const;
  at::ScalarType scalar_type;
  std::vector<int64_t> dims;
};

void validateTensorShapeAndType(torch::jit::Value *value,
                                const at::Tensor &tensor);

// setNodeTensorAttrValue and getNodeTensorAttrValue must be used instead of
// node->t_(c10::attr::value, v) and node->t(c10::attr::value).
//
// When printing a torch::jit::Graph the graph will iterate over each node
// and print all its attributes.
// If an attribute is a tensor it will try to print the content of that
// tensor which in our case would trigger a copy from IPU to CPU. This copy
// not only will fail, it will also be interpreted as a request to add this
// tensor as a graph output which will corrupt the graph.
// However attributes which are arrays of tensors are not printed and therefore
// will not trigger a copy, so behind the scenes these functions will wrap
// and unwrap the tensor attribute in a size 1 vector.
void setNodeTensorAttrValue(torch::jit::Node *node,
                            torch::jit::TensorAttr::ConstructorType value);
const torch::jit::TensorAttr::ValueType &
getNodeTensorAttrValue(const torch::jit::Node *node);
} // namespace poptorch

#endif // INCLUDE_POPTORCH_UTILS_HPP
