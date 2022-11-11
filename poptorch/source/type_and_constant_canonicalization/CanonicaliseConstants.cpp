// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <ATen/ATen.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include <any>
#include <functional>
#include <iterator>
#include <limits>
#include <stack>
#include <utility>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/DispatchTracer.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"

#include "../PoptorchSymbols.hpp"

namespace poptorch {
namespace type_and_constant_canonicalization {
namespace {

// Returns true for node kinds which change compiler state. These need to be
// removed for any host side tensors but otherwise does not make connected
// node a PopART only node.
bool compilerStateChangingKind(const torch::jit::NodeKind &kind) {
  return (kind == symbols::poptorch::begin_ipu_block ||
          kind == symbols::poptorch::end_ipu_block ||
          kind == symbols::poptorch::set_available_memory ||
          kind == symbols::poptorch::push_name_scope ||
          kind == symbols::poptorch::set_matmul_serialization);
}

bool popartOnlyNode(const torch::jit::NodeKind &kind) {
  return (!compilerStateChangingKind(kind) && kind != c10::prim::Constant &&
          kind != c10::prim::TupleConstruct &&
          kind != c10::prim::ListConstruct && kind != c10::prim::TupleUnpack &&
          kind != c10::prim::ListUnpack && kind != c10::prim::Return);
}

// Check whether the node is (eventually) used host side, IPU or both
UseOfNode getUseOfNode(torch::jit::Node *n,
                       bool check_node_kind_itself = true) {
  // Check the kind of the node itself (for when not called on a prim constant).
  // This could be disabled explicitly by the caller.
  if (check_node_kind_itself && popartOnlyNode(n->kind())) {
    return UseOfNode::PopARTOnly;
  }
  if (check_node_kind_itself && n->kind() == c10::prim::Return) {
    return UseOfNode::HostSideOnly;
  }

  bool popart_use = false;
  bool host_use = false;

  // Check all outputs
  std::vector<torch::jit::Node *> to_check;
  to_check.push_back(n);
  while (!to_check.empty()) {
    auto *cur_node = to_check.back();
    to_check.pop_back();
    for (auto *output : cur_node->outputs()) {
      for (auto use : output->uses()) {
        auto use_kind = use.user->kind();
        if (use_kind == c10::prim::Return) {
          // This must be host use as we have not reached an op which would be
          // run on popart yet.
          host_use = true;
        } else if (popartOnlyNode(use_kind) ||
                   use_kind == symbols::poptorch::set_available_memory ||
                   use_kind == symbols::poptorch::set_matmul_serialization) {
          popart_use = true;
        } else {
          // We only need to check the node further if it is neither returned
          // nor used by a Popart op
          to_check.push_back(use.user);
        }
      }
    }
  }

  if (!host_use && !popart_use) {
    // Some nodes such as begin_ipu_block will simply remove the tensor so make
    // it a default tensor_constant for simplicity.
    return UseOfNode::PopARTOnly;
  }

  if (host_use && popart_use) {
    return UseOfNode::HostSideAndPopART;
  }
  if (host_use) {
    return UseOfNode::HostSideOnly;
  }
  return UseOfNode::PopARTOnly;
}

void replaceWithConstantTensor(torch::jit::Graph *graph, torch::jit::Node *n,
                               const at::Tensor &t) {
  ERROR_ON(n->kind() != c10::prim::Constant);
  const bool is_dispatcher_active = isCompilingWithDispatcher();
  torch::jit::WithInsertPoint const insert_point(n);
  const WithNodeMetadata meta(n);

  poptorch::UseOfNode const use_of_node = getUseOfNode(n);
  auto *new_node = tensorToConstant(graph, t, use_of_node);

  if (!is_dispatcher_active) {
    // Due to tracing ambiguity, a float tensor here could be either
    // float or half
    auto new_type = new_node->output()->type()->expect<c10::TensorType>();
    if (new_type->scalarType() == at::ScalarType::Float) {
      new_node->output()->setType(new_type->withScalarType(HALF_OR_FLOAT));
    }
  }

  for (size_t use_idx = 0; use_idx < n->output()->uses().size(); use_idx++) {
    auto u = n->output()->uses()[use_idx];
    u.user->replaceInput(u.offset, new_node->output());
    use_idx--;
  }
}

void warnDoubleOutOfRange(double val, torch::jit::Node *n) {
  if (val > std::numeric_limits<float>::max() ||
      val < std::numeric_limits<float>::lowest()) {
    static std::uint64_t log_repeat = 0;
    logging::warn(log_repeat,
                  "{}: torch.float64 constant cannot be "
                  "represented as a torch.float32",
                  nodeToString(n));
  }
}

void warnLongOutOfRange(int64_t val, torch::jit::Node *n) {
  if (val > std::numeric_limits<int32_t>::max() ||
      val < std::numeric_limits<int32_t>::lowest()) {
    static std::uint64_t log_repeat = 0;
    logging::warn(log_repeat,
                  "{}: torch.int64 constant cannot be represented "
                  "as a torch.int32",
                  nodeToString(n));
  }
}

void handleNumberConstant(torch::jit::Graph *graph, torch::jit::Node *n) {
  if (n->output()->type()->isSubtypeOf(c10::BoolType::get())) {
    replaceWithConstantTensor(
        graph, n,
        at::native::scalar_tensor(*torch::jit::constant_as<bool>(n->output()),
                                  at::kInt, c10::nullopt, at::kCPU));
  } else {
    auto so = torch::jit::constant_as<at::Scalar>(n->output());
    ERROR_ON(!so.has_value());
    auto s = *so;

    c10::ScalarType dtype;
    if (s.isFloatingPoint()) {
      warnDoubleOutOfRange(s.toDouble(), n);
      dtype = at::kFloat;
    } else if (s.isIntegral(false)) {
      dtype = at::kInt;

      // Handle magic number 9223372036854775807
      if (s.toLong() == std::numeric_limits<int64_t>::max()) {
        s = std::numeric_limits<int32_t>::max();
        logging::info("{}: Using max value for torch.int32 in place of max "
                      "value for torch.int64",
                      nodeToString(n));
      } else {
        warnLongOutOfRange(s.toLong(), n);
      }
    } else {
      ERROR("Unsupported constant type");
    }

    auto wrapped_number =
        at::native::scalar_tensor(s, dtype, c10::nullopt, at::kCPU);
    wrapped_number.unsafeGetTensorImpl()->set_wrapped_number(true);
    replaceWithConstantTensor(graph, n, wrapped_number);
  }
}

void handleTensorConstant(torch::jit::Graph *graph, torch::jit::Node *n) {
  auto tensor_type = n->output()->type()->expect<c10::TensorType>();
  if (!tensor_type->scalarType().has_value()) {
    ERROR("Tensor constant without type");
  }

  at::Tensor tensor;
  if (n->kindOf(c10::attr::value) == torch::jit::AttributeKind::ts) {
    tensor = getNodeTensorAttrValue(n);
  } else {
    ERROR_ON_MSG(n->kindOf(c10::attr::value) != torch::jit::AttributeKind::t,
                 "[Internal] expected type 't' or 'ts' but got "
                     << torch::jit::toString(n->kindOf(c10::attr::value)));
    tensor = n->t(c10::attr::value);
  }
  ERROR_ON(!tensor.defined());
  const bool was_wrapped = tensor.unsafeGetTensorImpl()->is_wrapped_number();
  if (tensor.scalar_type() == at::ScalarType::Double) {
    warnDoubleOutOfRange(
        *reinterpret_cast<double *>(tensor.unsafeGetTensorImpl()->data()), n);
    tensor = tensor.to(at::ScalarType::Float);
  }
  if (tensor.scalar_type() == at::ScalarType::Long) {
    warnLongOutOfRange(
        *reinterpret_cast<int64_t *>(tensor.unsafeGetTensorImpl()->data()), n);

    tensor = tensor.to(at::ScalarType::Int);
  }

  // This gets lost in conversion and changes implicit casting if not set
  // (Must use an if as set_wrapped_number(false) can cause a PyTorch internal
  // error)
  if (was_wrapped) {
    tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  }

  replaceWithConstantTensor(graph, n, tensor);
}

void handleStringConstant(torch::jit::Graph *graph, torch::jit::Node *n) {
  std::string const s = n->s(c10::attr::value);
  std::vector<int64_t> shape_vec;
  shape_vec.push_back(s.length());

  auto t =
      at::empty({shape_vec}, at::dtype(at::ScalarType::Char)
                                 .memory_format(c10::MemoryFormat::Contiguous));

  std::memcpy(t.data_ptr(), s.c_str(), s.length() * sizeof(char));
  replaceWithConstantTensor(graph, n, t);
}

// Visit an ivalue which is a tuple or list constant and single type constant
// nodes and list/tuple constructs to replace it
class ListTupleVisitor {
  enum class State { IN_TUPLE, IN_LIST };

  // Maintain the information about the list or tuple at each level
  struct ListOrTupleInfo {
    ListOrTupleInfo(State state_, size_t elements_left_,
                    c10::TypePtr container_type_)
        : state(state_), elements_left(elements_left_),
          container_type(std::move(container_type_)) {}

    // Whether or not the visitor is currently in a list or a tuple
    State state;

    // The number of elenents left to be visited (before a List/TupleConstruct)
    size_t elements_left;

    // The type of the list/tuple, preserved from first visit ahead of
    // constructing the list or tuple
    c10::TypePtr container_type;

    // All the nodes to be input to the List/TupleConstruct
    std::vector<torch::jit::Node *> container_nodes;
  };

public:
  explicit ListTupleVisitor(torch::jit::Graph *graph)
      : _graph(graph), _last_node(nullptr) {}

  // We never return true as we visit every element
  bool operator()(const c10::IValue &i_value) {
    if (i_value.isGenericDict()) {
      ERROR("Dicts are not supported in constant canonicalisation.");
    }

    // Handle the visting of a list or tuple: actual creation will happen
    // once all its elements have been visited
    if (i_value.isTuple() || i_value.isList()) {
      handleListOrTuple(i_value);
      return false;
    }

    // Handle an element which is not a tuple or list
    handleConstant(i_value);

    // There will not be a further visit marking the completition of a tuple
    // or list, so this must be handled after the final constant.
    // In addition, in a nested scenario, this might trigger for then once
    // e.g. (1, (2, (3, 4))) will lead this block running three times.

    while (_info_stack.top().elements_left == 0) {
      handleTupleOrListConstruction();

      if (_info_stack.empty()) {
        // All tuples and lists have been constructed
        break;
      }
    }

    return false;
  }

  const std::vector<torch::jit::Node *> &getAllConstNodes() {
    return _all_const_nodes;
  }

  torch::jit::Node *getLastNode() {
    if (_last_node == nullptr) {
      // There is no last node: it means the list or tuple construction hasn't
      // been triggered (For example if it's an empty list/tuple).
      handleTupleOrListConstruction();
      ERROR_ON(_last_node == nullptr);
    }
    return _last_node;
  }

private:
  // Handle a list of the tuple: this involves merely recording the state, type
  // and number of elements as the inputs to a List/TupleConstruct will not have
  // been constructed at this point.
  void handleListOrTuple(const c10::IValue &i_value) {
    if (i_value.isTuple()) {
      _info_stack.emplace(State::IN_TUPLE, i_value.toTuple()->elements().size(),
                          i_value.type());
    } else {
      _info_stack.emplace(State::IN_LIST, i_value.toListRef().size(),
                          i_value.type());
    }
  }

  // Handle a tensor or numeric constant. This adds a constant of the same type
  // to the graph, which will later be canonicalised to a tensor constant.
  // Though this means that there will be an extra canonicalisation step, it
  // minimises code duplication. All constants are added to "_all_const_nodes"
  // for the later canonicalisation.
  void handleConstant(const c10::IValue &i_value) {
    ERROR_ON(_info_stack.empty());

    auto *new_const = _graph->create(c10::prim::Constant);

    if (i_value.isTensor()) {
      new_const->output()->inferTypeFrom(i_value.toTensor());
      setNodeTensorAttrValue(new_const, i_value.toTensor());
    } else if (i_value.isInt()) {
      new_const->output()->setType(c10::IntType::get());
      new_const->i_(c10::attr::value, i_value.toInt());
    } else if (i_value.isDouble()) {
      new_const->output()->setType(c10::FloatType::get());
      new_const->f_(c10::attr::value, i_value.toDouble());
    } else if (i_value.isBool()) {
      new_const->output()->setType(c10::BoolType::get());
      new_const->i_(
          c10::attr::value,
          static_cast<torch::jit::IntAttr::ConstructorType>(i_value.toBool()));
    } else if (i_value.isNone()) {
      // Assign NoneType so that the node can be skipped over
      // during constant canonicalization
      new_const->output()->setType(c10::NoneType::get());
    } else {
      ERROR("Unsupported type for constant: " << i_value);
    }

    insertNodeInGraph(_graph, new_const);
    _info_stack.top().container_nodes.push_back(new_const);
    _all_const_nodes.push_back(new_const);
    _info_stack.top().elements_left--;
  }

  // Handle the actual constructions of a list or tuple once the last element
  // has been visited.
  void handleTupleOrListConstruction() {
    torch::jit::Node *construct_node;

    switch (_info_stack.top().state) {
    case State::IN_TUPLE:
      construct_node = _graph->create(c10::prim::TupleConstruct);
      break;
    case State::IN_LIST:
      construct_node = _graph->create(c10::prim::ListConstruct);
      break;
    default:
      ERROR("Unreachable");
    }

    for (auto *element : _info_stack.top().container_nodes) {
      construct_node->addInput(element->output());
    }
    construct_node->output()->setType(_info_stack.top().container_type);
    insertNodeInGraph(_graph, construct_node);

    _info_stack.pop();

    if (!_info_stack.empty()) {
      ERROR_ON(_info_stack.top().elements_left < 1);
      _info_stack.top().elements_left--;

      // The container is itself an element of the previous container
      _info_stack.top().container_nodes.push_back(construct_node);
    } else {
      // Store the final node for access outside the visit
      _last_node = construct_node;
    }
  }

  torch::jit::Graph *_graph;
  std::stack<ListOrTupleInfo> _info_stack;
  std::vector<torch::jit::Node *> _all_const_nodes;
  torch::jit::Node *_last_node;
};

void handleListOrTuple(torch::jit::Graph *graph, torch::jit::Node *n,
                       std::unordered_set<torch::jit::Node *> *to_delete) {
  torch::jit::WithInsertPoint const insert_point(n);
  const WithNodeMetadata meta(n);

  // Use the visitor to turn the single list/tuple constant into many
  // constants and List/TupleConstructs.
  ListTupleVisitor visitor(graph);
  const auto &tuple_ivalue = n->ival(c10::attr::value);
  tuple_ivalue.visit(std::function<bool(const c10::IValue &)>(
      std::reference_wrapper(visitor)));

  // Find the very last node added and use it to replace the original node
  auto *replacement_node = visitor.getLastNode();
  auto *replacement_node_out = replacement_node->output();
  replacement_node_out->setType(n->output()->type());
  n->output()->replaceAllUsesWith(replacement_node_out);

  // The nodes added in the visitor match those of constants not in lists/tuples
  // *before* canonicalisation (to permit code reuse). Hence, we canonicalise
  // in the same way.
  for (auto *prim_const : visitor.getAllConstNodes()) {
    torch::jit::WithInsertPoint const insert_point_prim_const(prim_const);
    const WithNodeMetadata prim_meta(prim_const);

    // If there are NoneTypes we can skip those
    if (prim_const->output()->type() != c10::NoneType::get()) {
      if (prim_const->output()->type()->isSubtypeOf(c10::TensorType::get())) {
        handleTensorConstant(graph, prim_const);
      } else {
        handleNumberConstant(graph, prim_const);
      }
      to_delete->insert(prim_const);
    }
  }
}

void recursivelySelectHostAndIPUSideConstants(
    torch::jit::Node *node_to_process, torch::jit::Node *host_side_replacement,
    torch::jit::Node *ipu_side_replacement,
    std::unordered_set<torch::jit::Node *> *to_delete) {
  for (size_t output_idx = 0; output_idx < node_to_process->outputs().size();
       output_idx++) {
    auto *output = node_to_process->output(output_idx);

    while (!output->uses().empty()) {
      auto use = output->uses()[0];
      switch (getUseOfNode(use.user)) {
      case UseOfNode::HostSideOnly:
        use.user->replaceInput(use.offset,
                               host_side_replacement->output(output_idx));
        break;
      case UseOfNode::PopARTOnly:
        use.user->replaceInput(use.offset,
                               ipu_side_replacement->output(output_idx));
        break;
      case UseOfNode::HostSideAndPopART:
        auto *graph = use.user->owningGraph();
        torch::jit::WithInsertPoint const insert_point(use.user);
        const WithNodeMetadata meta(use.user);

        auto same_value = [](torch::jit::Value *value) { return value; };

        auto *host_side_node = graph->createClone(use.user, same_value);
        host_side_node->replaceInput(use.offset,
                                     host_side_replacement->output(output_idx));

        insertNodeInGraph(graph, host_side_node);

        auto *ipu_side_node = graph->createClone(use.user, same_value);
        ipu_side_node->replaceInput(use.offset,
                                    ipu_side_replacement->output(output_idx));
        insertNodeInGraph(graph, ipu_side_node);

        recursivelySelectHostAndIPUSideConstants(use.user, host_side_node,
                                                 ipu_side_node, to_delete);

        to_delete->insert(use.user);

        // Prevent further cloning
        while (!use.user->inputs().empty()) {
          use.user->removeInput(0);
        }
        break;
      }
    }
  }
}

// Find any host_and_ipu_side_tensor_constant constants and perform the
// necessary splitting
void rectifyHostAndIPUSideConstants(
    torch::jit::Graph *graph,
    std::unordered_set<torch::jit::Node *> *to_delete) {
  logging::LogContext const ctx_func("rectifyHostAndIPUSideConstants");
  for (auto *node : graph->nodes()) {
    logging::LogContext const ctx("processing " + nodeToString(node));

    if (node->kind() != symbols::poptorch::host_and_ipu_side_tensor_constant) {
      continue;
    }

    // Create two new nodes
    auto t = getNodeTensorAttrValue(node);
    torch::jit::WithInsertPoint const insert_point(node);
    const WithNodeMetadata meta(node);

    torch::jit::Node *host_side_node = createAndInsertNode(
        graph, symbols::poptorch::host_side_tensor_constant);
    host_side_node->output()->inferTypeFrom(t);
    setNodeTensorAttrValue(host_side_node, t);

    torch::jit::Node *ipu_node =
        createAndInsertNode(graph, symbols::poptorch::tensor_constant);
    ipu_node->output()->inferTypeFrom(t);
    setNodeTensorAttrValue(ipu_node, t);

    recursivelySelectHostAndIPUSideConstants(node, host_side_node, ipu_node,
                                             to_delete);

    to_delete->insert(node);
  }
}

void removeStateChangingNodesFromHostSideBranch(
    torch::jit::Graph *graph,
    std::unordered_set<torch::jit::Node *> *to_delete) {
  logging::LogContext const ctx_func(
      "removeStateChangingNodesFromHostSideBranch");
  for (auto *node : graph->nodes()) {
    logging::LogContext const ctx("processsing " + nodeToString(node));
    if (node->kind() != symbols::poptorch::host_side_tensor_constant) {
      continue;
    }

    std::vector<torch::jit::Node *> to_process;
    to_process.push_back(node);
    while (!to_process.empty()) {
      auto *cur_node = to_process.back();
      to_process.pop_back();

      auto outputs = cur_node->outputs();
      for (auto *output : outputs) {
        for (auto use : output->uses()) {
          to_process.push_back(use.user);
        }
      }

      if (!compilerStateChangingKind(cur_node->kind())) {
        continue;
      }

      // The number of outputs may be less e.g. begin_ipu_block, but otherwise
      // any output to be replaced must match the input for this to work.
      for (size_t output_idx = 0; output_idx < cur_node->outputs().size();
           output_idx++) {
        cur_node->output(output_idx)
            ->replaceAllUsesWith(cur_node->input(output_idx));
      }

      to_delete->insert(cur_node);
    }
  }
}

void canonicaliseIfConstant(torch::jit::Graph *graph, torch::jit::Node *node,
                            std::unordered_set<torch::jit::Node *> *to_delete) {
  logging::LogContext const ctx("processing " + nodeToString(node));

  if (node->kind() == c10::aten::size) {
    // This will be made a constant in the size handler
    node->output()->setType(
        c10::TensorType::create(c10::ScalarType::Int, c10::nullopt, 1, false));
  }

  // If it's not a constant or if it doesn't have a value (i.e is None) or if
  // it's a Device
  if (node->kind() != c10::prim::Constant ||
      !node->hasAttribute(c10::attr::value) ||
      node->output()->type()->isSubtypeOf(c10::DeviceObjType::get())) {
    return;
  }

  if (node->output()->type()->isSubtypeOf(c10::NumberType::get()) ||
      node->output()->type()->isSubtypeOf(c10::BoolType::get())) {
    logging::LogContext const ctx2("handling as number constant");
    handleNumberConstant(graph, node);
  } else if (node->output()->type()->isSubtypeOf(c10::TensorType::get())) {
    logging::LogContext const ctx2("handling as tensor constant");
    handleTensorConstant(graph, node);
  } else if (node->output()->type()->isSubtypeOf(c10::StringType::get())) {
    logging::LogContext const ctx2("handling as string constant");
    handleStringConstant(graph, node);
  } else if (node->output()->type()->isSubtypeOf(c10::ListType::ofBools())) {
    // Only known case is the result of an evaluated constexpr
    logging::LogContext const ctx2("handling as bool list constant");
    handleListOrTuple(graph, node, to_delete);
  } else if (node->output()->type()->isSubtypeOf(c10::ListType::ofFloats())) {
    // Only known case is the result of an evaluated constexpr
    logging::LogContext const ctx2("handling as float list constant");
    handleListOrTuple(graph, node, to_delete);
  } else if (node->output()->type()->isSubtypeOf(c10::ListType::ofInts())) {
    // Only known case is the result of an evaluated constexpr
    logging::LogContext const ctx2("handling as int list constant");
    handleListOrTuple(graph, node, to_delete);
  } else if (node->output()->type()->isSubtypeOf(c10::ListType::ofTensors())) {
    // Only known case is the result of an evaluated constexpr
    logging::LogContext const ctx2("handling a tensor list constant");
    handleListOrTuple(graph, node, to_delete);
  } else if (node->output()->type()->isSubtypeOf(c10::ListType::create(
                 c10::OptionalType::create(c10::TensorType::get())))) {
    logging::LogContext const ctx2("handling an optional tensor list constant");
    handleListOrTuple(graph, node, to_delete);
  } else if (node->output()->type()->cast<c10::TupleType>()) {
    handleListOrTuple(graph, node, to_delete);
  } else {
    ERROR("Unsupported type " << node->output()->type()->str());
  }

  to_delete->insert(node);
}

void convertReturnedInputsToConstants(
    torch::jit::Graph *graph, std::vector<std::size_t> &input_index_map) {
  std::stack<std::size_t> to_erase;
  for (auto i = 0u; i < graph->inputs().size(); i++) {
    auto *input = graph->inputs()[i];
    if (input->uses().size() == 1 &&
        input->uses()[0].user->kind() == c10::prim::Return) {
      const WithNodeMetadata meta(input->node());
      torch::jit::Value *new_const =
          insertConstant(graph, input->node()->ts(c10::attr::values)[i]);
      input->replaceAllUsesWith(new_const);
      to_erase.push(i);
    } else {
      // Save the mapping from PopART input indices to user input indices,
      // so that the returned-only inputs can be ignored by PopART
      input_index_map.push_back(i);
    }
  }

  while (!to_erase.empty()) {
    graph->eraseInput(to_erase.top());
    to_erase.pop();
  }
}

} // namespace

void canonicaliseConstants(torch::jit::Graph *graph,
                           std::vector<std::size_t> &input_index_map) {
  logging::LogContext const ctx_func("CanonicaliseConstants");
  std::unordered_set<torch::jit::Node *> to_delete;

  if (isCompilingWithDispatcher()) {
    // If any inputs are simply returned as outputs, replace
    // those inputs with host-side-only constants so that they
    // aren't lowered to PopART
    convertReturnedInputsToConstants(graph, input_index_map);
  }

  for (auto *node : graph->nodes()) {
    canonicaliseIfConstant(graph, node, &to_delete);
  }

  searchAndPossiblyDestroy(to_delete);
  to_delete.clear();

  rectifyHostAndIPUSideConstants(graph, &to_delete);
  searchAndPossiblyDestroy(to_delete);

  to_delete.clear();
  removeStateChangingNodesFromHostSideBranch(graph, &to_delete);
  searchAndPossiblyDestroy(to_delete);
}

} // namespace type_and_constant_canonicalization
} // namespace poptorch
