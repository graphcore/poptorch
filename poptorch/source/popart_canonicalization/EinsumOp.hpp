// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <torch/csrc/jit/ir/ir.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace poptorch {
class EinsumOp {
public:
  EinsumOp(std::string eq, const std::vector<torch::jit::Value *> &tensors);

  torch::jit::Node *create(torch::jit::Graph *graph);

private:
  // A modified version of tensordot that handles batch dimensions and takes
  // two tensors of the same rank that have been unsqueezed (if necessary) to
  // match. The output is of the same rank. Batch dims always appear first in
  // the output to allow chaining.
  torch::jit::Node *tensordotBmm(torch::jit::Graph *graph,
                                 torch::jit::Value *x1,
                                 torch::jit::Value *x2) const;

  // Get permute indices of 's' according to the order specified by char_indices
  template <typename T>
  std::vector<std::int64_t>
  sortedPermutation(const std::unordered_map<char, std::size_t> &char_indices,
                    const T &s) const {
    std::vector<std::int64_t> p(s.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](auto d1, auto d2) {
      return char_indices.at(s[d1]) < char_indices.at(s[d2]);
    });
    return p;
  }

  // Ensure all tensors have same number of dims that are in the same order -
  // The order in which they appear in the lhs
  void canonicalizeTensors(torch::jit::Graph *graph);

  // This combines permuting batch dims to their original locations, and
  // then permuting to the order specified by the rhs
  std::vector<std::int64_t> getOutputPermutation() const;

  std::vector<torch::jit::Value *> _tensors;
  std::string _lhs, _rhs;
  std::vector<std::string> _labels;
  std::size_t _n_dims;
  // List of characters ordered as seen from left to right. This
  // is the order of dims during the multiply/reduce stage
  std::vector<char> _ordered_chars;
  // Used to determine whether a non-reduce dimension should be
  // considered a batch dimension during calculation
  std::unordered_map<char, int> _char_counts_seen;
  // Number of times a character appears in future operands -
  // used to determine whether a dimension should be reduced
  std::unordered_map<char, int> _char_counts_remaining;
  // Mapping of each character to the index in which it appears
  // in the intermediate tensor shape
  std::unordered_map<char, std::size_t> _lhs_char_indices;
  // Mapping of each character to the index in which it appears
  // in the output shape
  std::unordered_map<char, std::size_t> _rhs_char_indices;
  // Bitset indicating dimensions to be reduced
  std::vector<bool> _rdims_bs;
  // Bitset indicating batch dimensions
  std::vector<bool> _bdims_bs;
  // Bitset indicating dimensions that appear in rhs
  std::vector<bool> _rhs_bs;
}; // class einsum
} // namespace poptorch
