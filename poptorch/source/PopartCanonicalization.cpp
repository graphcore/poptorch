// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <torch/csrc/jit/ir/ir.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "PoptorchSymbols.hpp"
#include "popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "poptorch/DispatchTracer.hpp"
#include "poptorch/InplaceOps.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace torch {
namespace jit {
bool isInplaceOp(const Node *node);
} // namespace jit
} // namespace torch

namespace poptorch {
namespace {

// In-place modification of slices is a special case. When we
// modify a slice in-place, torch produces a graph like the
// following:
//
//   %x = input, shape = [4, 4]
//   %1 = slice(%x), shape = [2, 2]
//   %2 = add(%1, %1)
//   %3 = slice(%x), shape = [2, 2]
//   %4 = copy_(%3, %2), shape = [2, 2]
//   return %x, shape = [4, 4]
//
// The original input %x is returned because the slice %3 is a
// view on %x, and thus any modifications to %3 are reflected
// in %x. To simulate in-place modification to slices, we return
// a dynamic update instead, so that we can perform the slice
// modification out-of-place, and return the "modified" tensor
// with the correct shape
//
//   %x = input, shape = [4, 4]
//   %1 = slice(%x), shape = [2, 2]
//   %2 = add(%1, %1)
//   %3 = dynamic_update(%x, $2) shape = [4, 4]
//   return %3, shape = [4, 4]
//
torch::jit::Node *handleSliceModification(torch::jit::Graph *graph,
                                          torch::jit::Node *node,
                                          torch::jit::Value *modified_slice) {
  torch::jit::Value *input = node->input(0);
  torch::jit::Node *new_node = modified_slice->node();
  // Follow the chain of slices that are being operated on by the inplace op
  while (input->node()->kind() == symbols::popart::slice) {
    auto *slice = input->node();
    auto *slice_input = slice->input(0);

    // Record the indices that we sliced: We need these for DynamicUpdate
    std::vector<int64_t> slice_starts = slice->is(c10::Symbol::attr("starts"));
    std::vector<int64_t> slice_ends = slice->is(c10::Symbol::attr("ends"));
    std::vector<int64_t> slice_dims = slice->is(c10::Symbol::attr("axes"));

    auto *slice_offset =
        createConstantInt(graph, slice_starts,
                          {static_cast<int64_t>(slice_starts.size())})
            ->output();

    std::vector<int64_t> sizes(slice_starts.size());
    std::transform(std::begin(slice_ends), std::end(slice_ends),
                   std::begin(slice_starts), std::begin(sizes),
                   std::minus<int64_t>());

    auto *dynamic_update =
        createDynamicupdate(graph, {slice_input, slice_offset, modified_slice},
                            slice_dims, sizes, /* noOverlap = */ 1);

    // Replace uses of slice input after inplace op with result of
    // the dynamic update (i.e. the modified tensor)
    auto *modified_input = dynamic_update->output();
    slice_input->replaceAllUsesAfterNodeWith(node, modified_input);
    new_node = dynamic_update;

    // Repeat this process for the entire chain of slices - the
    // reconstructed modified input is used to reconstruct the next
    // modified slice input
    input = slice_input;
    modified_slice = modified_input;
  }
  // Dynamic update does not support step size. Slicing with step size is
  // implemented using subsample(slice(x))
  if (input->node()->kind() == symbols::popart::subsample) {
    auto *subsample = input->node();
    if (subsample->input(0)->node()->kind() == symbols::popart::slice) {
      logging::warn(
          "In-place modification of slices with step size other than 1 is "
          "not supported. This may result in unexpected behaviour.");
    }
  }
  return new_node;
}

// Propagates half types across lists (tuple set to false) or tuples (tuple set
// to true).
// If the new node is a List/TupleConstruct, it will not, by default, have the
// types set to half when they should be, because tracing is always performed
// with floats. Use this function to rememby that on a List/Tuple construct
// after it has been created.
void propagateHalfOnListOrTupleConstruct(torch::jit::Node *n, bool tuple) {
  auto constr_type = tuple ? at::prim::TupleConstruct : at::prim::ListConstruct;
  auto unpack_type = tuple ? at::prim::TupleUnpack : at::prim::ListUnpack;

  if (n->kind() != constr_type) {
    return;
  }

  // Record which inputs were half: they would not have been on tracing but
  // would be change during canonicalization
  std::vector<bool> input_was_half;
  input_was_half.reserve(n->inputs().size());
  for (auto *input : n->inputs()) {
    // Skip if it is not a tensor or has no scalar type
    auto tensor_type = input->type()->cast<c10::TensorType>();
    if ((!tensor_type) || !tensor_type->scalarType()) {
      input_was_half.emplace_back(false);
      continue;
    }

    input_was_half.emplace_back(getNodeScalarType(input) ==
                                at::ScalarType::Half);
  }

  // Propagate types on the unpack node(s)
  for (const auto &use : n->output()->uses()) {
    torch::jit::Node *unpack = use.user;
    if (unpack->kind() != unpack_type) {
      continue;
    }

    size_t idx = 0;
    for (auto *output : unpack->outputs()) {
      // The output will be float as tracing was carried out using floats.
      if (input_was_half[idx]) {
        output->setType(
            output->type()->expect<c10::TensorType>()->withScalarType(
                c10::ScalarType::Half));
      }
      idx++;
    }
  }
}

class CanonicalizeImpl {
public:
  static void run(torch::jit::Graph *graph);
};

/*
 * ConvertAtenToPopart implementation.
 */

void CanonicalizeImpl::run(torch::jit::Graph *graph) {
  logging::LogContext ctx_func("PopartCanonicalization");
  for (torch::jit::Node *node : graph->nodes()) {
    logging::LogContext ctx("processing " + nodeToString(node));
    WithNodeMetadata metadata(node);
    torch::jit::WithInsertPoint insert_point(node);
    torch::jit::Node *new_node = nullptr;
    torch::jit::Symbol kind = node->kind();

    if (SymbolHandler handler = getHandler(kind)) {
      new_node = handler(graph, node);

      const bool was_inplace_op_on_view =
          node->hasAttributeS("was_inplace_on_view") &&
          node->i(c10::Symbol::attr("was_inplace_on_view")) == 1;

      if (was_inplace_op_on_view || torch::jit::isInplaceOp(node)) {
        new_node = handleSliceModification(graph, node, new_node->output());
      }
    }

    // If we have a new node add it and replace the old use.
    if (new_node != nullptr) {
      // Mark this node for deletion.
      markNodeForDeletion(node);

      if (node->hasUses()) {
        for (std::uint64_t i = 0; i < node->outputs().size(); ++i) {
          if (i >= new_node->outputs().size()) {
            ERROR_ON_MSG(
                node->output(i)->hasUses(),
                "The canonicalised JIT node has fewer outputs than the "
                "dispatch function. This is only an issue because these "
                "outputs are used.");
            continue;
          }

          // As well as replacing the use, this will copy across shape/type
          // if not explicitly set.
          replaceOutputUse(node, new_node, i);
        }

        // Propagate half types across ListConstructs and TupleConstructs
        propagateHalfOnListOrTupleConstruct(new_node, true);
        propagateHalfOnListOrTupleConstruct(new_node, false);
      }
    }
  }

  // Build a list of nodes marked for deletion.
  std::unordered_set<torch::jit::Node *> to_delete;
  for (torch::jit::Node *node : graph->nodes()) {
    if (isMarkedForDeletion(node)) {
      to_delete.insert(node);
    }
  }

  // Remove the dead nodes.
  searchAndPossiblyDestroy(to_delete);
}

} // namespace

void canonicalize(torch::jit::Graph *graph) {
  CanonicalizeImpl converter;
  converter.run(graph);
}
} // namespace poptorch
