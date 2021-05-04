// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "EinsumOp.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {

EinsumOp::EinsumOp(std::string eq,
                   const std::vector<torch::jit::Value *> &tensors) {
  _tensors = tensors;
  // Remove all whitespace in equation
  eq.erase(std::remove(eq.begin(), eq.end(), ' '), eq.end());
  _lhs = eq;

  auto pos = eq.find("->");
  if (pos != std::string::npos) {
    _lhs = eq.substr(0, pos);
    // Add 2 to exclude arrow
    _rhs = eq.substr(pos + 2);
  }

  // Split lhs into labels using ',' delimiter
  std::stringstream ss(_lhs);
  std::string s;
  while (std::getline(ss, s, ',')) {
    _labels.push_back(s);
  }
  ERROR_ON(_labels.size() != _tensors.size());

  for (const auto &label : _labels) {
    for (char c : label) {
      if (_lhs_char_indices.find(c) == _lhs_char_indices.end()) {
        _lhs_char_indices[c] = _ordered_chars.size();
        _char_counts_seen[c] = 0;
        _char_counts_remaining[c] = 1;
        _ordered_chars.push_back(c);
      } else {
        _char_counts_remaining[c]++;
      }
    }
  }
  // Shared rank of tensors during multiplication
  _n_dims = _ordered_chars.size();

  // Calculate implicit rhs according to classical einstein summation
  if (pos == std::string::npos) {
    std::copy_if(_ordered_chars.begin(), _ordered_chars.end(),
                 std::back_inserter(_rhs),
                 [&](char c) { return _char_counts_remaining[c] == 1; });
    // Must be alphabetical in this case
    std::sort(_rhs.begin(), _rhs.end());
  }

  _rdims_bs.resize(_n_dims);
  _bdims_bs.resize(_n_dims);
  _rhs_bs.resize(_n_dims);
  for (char c : _rhs) {
    _rhs_bs[_lhs_char_indices[c]] = true;
    _rhs_char_indices[c] = _rhs_char_indices.size();
  }
  // All characters must be present in the map but only the indices of rhs
  // characters matter
  for (char c : _lhs) {
    _rhs_char_indices.emplace(c, 0);
  }
}

torch::jit::Node *
EinsumOp::create(torch::jit::Graph *graph,
                 const std::vector<std::int64_t> &output_shape) {
  canonicalizeTensors(graph);

  torch::jit::Node *output = nullptr;

  // One tensor means only a summation is applied
  if (_tensors.size() == 1) {
    std::vector<std::int64_t> axes;

    for (std::size_t i = 0; i < _n_dims; i++) {
      if (!_rhs_bs[i]) {
        axes.push_back(static_cast<std::int64_t>(i));
      }
    }
    output = createReducesum(graph, {_tensors[0]}, axes, 1);
  } else {
    updateCharCounts(_labels[0]);
    // Base output
    output = _tensors[0]->node();
    // Build product from left to right
    for (std::size_t i = 1; i < _tensors.size(); i++) {
      output = createProduct(graph, output->output(), _tensors[i], _labels[i]);
    }
    output = permuteOutput(graph, output->output());
  }

  // Remove reduced single dimensions by reshaping
  return createReshape(graph, output->output(), output_shape);
}

torch::jit::Node *EinsumOp::tensordotBmm(torch::jit::Graph *graph,
                                         torch::jit::Value *x1,
                                         torch::jit::Value *x2) const {
  const std::vector<std::int64_t> shape_x1 = shapeFromTensor(x1);
  const std::vector<std::int64_t> shape_x2 = shapeFromTensor(x2);
  ERROR_ON(shape_x1.size() != shape_x2.size());

  std::int64_t rdims_prod = 1;
  std::int64_t bdims_prod = 1;
  for (std::size_t i = 0; i < _n_dims; i++) {
    if (_rdims_bs[i]) {
      if (shape_x1[i] == shape_x2[i]) {
        rdims_prod *= shape_x1[i];
      } else if (shape_x1[i] == 1) {
        x2 = createReducesum(graph, {x2}, {static_cast<std::int64_t>(i)}, 1)
                 ->output();
      } else if (shape_x2[i] == 1) {
        x1 = createReducesum(graph, {x1}, {static_cast<std::int64_t>(i)}, 1)
                 ->output();
      }
    }
    if (_bdims_bs[i]) {
      bdims_prod *= shape_x1[i];
    }
  }

  // Partitions existing permutation vector p according to bitset bs. If
  // should_partition_front == true, elements of p are moved to the front
  // if the corresponding bool in bs == true. Otherwise, they are moved to
  // the back. The relative order of other elements must not change.
  auto fn_partition = [&](auto &p, const auto &bs,
                          bool should_partition_front) {
    std::stable_partition(p.begin(), p.end(), [&](std::int64_t n) {
      return bs[n] == should_partition_front;
    });
  };

  // Original permutation
  std::vector<std::int64_t> p1(_n_dims);
  std::iota(p1.begin(), p1.end(), 0);
  std::vector<std::int64_t> p2 = p1;

  // Cast the reduction to a batch matrix multiplication by permuting input
  // dimensions and reshaping to ensure there is one batch dimension and
  // one reduce (dot product) dimension.

  // Permute x1 so that rdims are the last dims
  fn_partition(p1, _rdims_bs, false);
  // Permute again so that bdims are the first dims
  fn_partition(p1, _bdims_bs, true);
  torch::jit::Node *p_x1 = createTranspose(graph, {x1}, p1);
  // Reshape to (bdims_prod, -1, rdims_prod)
  torch::jit::Node *p_x1_bmat =
      createReshape(graph, p_x1->output(), {bdims_prod, -1, rdims_prod});

  // Permute x2 so that rdims are the first dims
  fn_partition(p2, _rdims_bs, true);
  // Permute again so that bdims are the first dims and rdims follow
  fn_partition(p2, _bdims_bs, true);
  torch::jit::Node *p_x2 = createTranspose(graph, {x2}, p2);
  // Reshape to (bdims_prod, rdims_prod, -1)
  torch::jit::Node *p_x2_bmat =
      createReshape(graph, p_x2->output(), {bdims_prod, rdims_prod, -1});

  // Matmul -> (bdims_prod, unreduced_x1, unreduced_x2)
  torch::jit::Node *mm =
      createMatmul(graph, {p_x1_bmat->output(), p_x2_bmat->output()});

  std::vector<std::int64_t> new_shape;
  for (std::size_t i = 0; i < _n_dims; i++) {
    if (_bdims_bs[i]) {
      new_shape.push_back(shape_x1[i]);
    }
  }
  for (std::size_t i = 0; i < _n_dims; i++) {
    if (_rdims_bs[i]) {
      new_shape.push_back(1);
    } else if (!_bdims_bs[i]) {
      // If not a batch dim or reduce dim, at least one dim == 1
      // so we can multiply to get the right result
      new_shape.push_back(shape_x1[i] * shape_x2[i]);
    }
  }

  // Restore flattened dims
  return createReshape(graph, mm->output(), new_shape);
}

void EinsumOp::canonicalizeTensors(torch::jit::Graph *graph) {
  for (std::size_t i = 0; i < _tensors.size(); i++) {
    torch::jit::Value *t = _tensors[i];
    std::vector<std::int64_t> shape = shapeFromTensor(t);

    // Get permute indices of lhs
    std::vector<std::int64_t> p_lhs =
        sortedPermutation(_lhs_char_indices, _labels[i]);

    // Calculate permuted shape and label
    std::vector<std::int64_t> shape_p;
    std::transform(p_lhs.begin(), p_lhs.end(), std::back_inserter(shape_p),
                   [&](auto d) { return shape[d]; });

    // TODO(T6451): Implement diagonals whenever ai.onnx.EyeLike is implemented
    //              in PopART

    // Insert missing dims
    for (std::size_t j = 0; j < _ordered_chars.size(); j++) {
      if (_labels[i].find(_ordered_chars[j]) == std::string::npos) {
        shape_p.insert(shape_p.begin() + j, 1);
      }
    }

    // Permute and reshape
    t = createTranspose(graph, {t}, p_lhs)->output();
    _tensors[i] = createReshape(graph, t, shape_p)->output();
  }
}

torch::jit::Node *EinsumOp::permuteOutput(torch::jit::Graph *graph,
                                          torch::jit::Value *output) const {
  std::vector<char> out_chars = _ordered_chars;
  std::stable_partition(out_chars.begin(), out_chars.end(), [&](char c) {
    return _bdims_bs[_lhs_char_indices.at(c)];
  });

  // Permute batch dims back to original locations
  std::vector<std::int64_t> p_lhs =
      sortedPermutation(_lhs_char_indices, out_chars);

  // Permute to the order specified by rhs
  std::vector<std::int64_t> p_rhs =
      sortedPermutation(_rhs_char_indices, _ordered_chars);

  // Combine permutations
  std::vector<std::int64_t> p_combined;
  std::transform(p_rhs.begin(), p_rhs.end(), std::back_inserter(p_combined),
                 [&](auto d) { return p_lhs[d]; });

  return createTranspose(graph, {output}, p_combined);
}

void EinsumOp::updateCharCounts(const std::string &label) {
  for (char c : label) {
    _char_counts_seen[c]++;
    _char_counts_remaining[c]--;
  }
}

torch::jit::Node *EinsumOp::createProduct(torch::jit::Graph *graph,
                                          torch::jit::Value *lhs,
                                          torch::jit::Value *rhs,
                                          const std::string &rhs_label) {
  updateCharCounts(rhs_label);

  for (std::size_t i = 0; i < _n_dims; i++) {
    char c = _ordered_chars[i];
    // if dim appears in rhs, don't reduce
    // if dim appears in future operands, don't reduce yet
    _rdims_bs[i] =
        !_rhs_bs[i] && _char_counts_remaining[_ordered_chars[i]] == 0;
    _bdims_bs[i] = !_rdims_bs[i] && _char_counts_seen[c] > 1;
  }

  return tensordotBmm(graph, lhs, rhs);
}
} // namespace poptorch
