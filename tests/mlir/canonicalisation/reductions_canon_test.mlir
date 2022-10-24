// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// RUN: poptorch-opt --canonicalize %s --split-input-file | FileCheck %s


// Test case: `max_dim` of a zero-sized tensor (ie. wrapped scalar) -> tuple
// (`clone(input)`, `zeros_like(result[0])`)
//
// (the `zeros_like` will then canonicalise to a `full`)

func.func @max_dim_scalar(%input: tensor<si32>) -> (tensor<si32>, tensor<1xsi32>) {
  %vals, %idxs = "poptorch.max_dim"(%input) {dim = 0, keepdim = false} : (tensor<si32>) -> (tensor<si32>, tensor<1xsi32>)

  // Write to the value-result, so that the `clone` doesn't get removed too.
  "poptorch.overwrite"(%vals, %input) : (tensor<si32>, tensor<si32>) -> ()

  return %vals, %idxs : tensor<si32>, tensor<1xsi32>
}

// CHECK-LABEL: @max_dim_scalar(
// CHECK-SAME:                  %[[INPUT:.*]]: tensor<si32>
// CHECK-NOT: poptorch.max_dim
// CHECK: %[[VALS:.*]] = "poptorch.clone"(%[[INPUT]]) : (tensor<si32>) -> tensor<si32>
// CHECK-NOT: poptorch.max_dim
// CHECK: %[[IDXS:.*]] = "poptorch.full"() {fill_value = 0
// CHECK-SAME:                                            } : () -> tensor<1xsi32>
// CHECK-NOT: poptorch.max_dim
// CHECK: return %[[VALS]], %[[IDXS]] : tensor<si32>, tensor<1xsi32>


// -----

// Test case: `max_dim(keep_dim=false)` should canon. to `topk(1)`, with the
// output values & indices then going through `squeeze_dim`.
//
// (the `squeeze_dim` will then canonicalise to a `view`)

func.func @max_dim_nokeepdim(%input: tensor<2x3xsi32>) -> (tensor<3xsi32>, tensor<3xsi32>) {
  %vals, %idxs = "poptorch.max_dim"(%input) {dim = 0, keepdim = false} : (tensor<2x3xsi32>) -> (tensor<3xsi32>, tensor<3xsi32>)
  return %vals, %idxs : tensor<3xsi32>, tensor<3xsi32>
}

// CHECK-LABEL: @max_dim_nokeepdim(
// CHECK-SAME:                     %[[INPUT:.*]]: tensor<2x3xsi32>
// CHECK-NOT: poptorch.max_dim
// CHECK: %[[VALS:.*]], %[[IDXS:.*]] = "poptorch.topk"(%[[INPUT]]) {
// CHECK-SAME:                                                      K = 1
// CHECK-SAME:                                                      dim = 0
// CHECK-SAME:                                                      largest = true
// CHECK-SAME:                                                      sorted = false
// CHECK-SAME:                                                     } : (tensor<2x3xsi32>) -> (tensor<1x3xsi32>, tensor<1x3xsi32>)
// CHECK-NOT: poptorch.max_dim
// CHECK: %[[SQUEEZED_VALS:.*]] = "poptorch.view"(%[[VALS]]) {shape = [3]} : (tensor<1x3xsi32>) -> tensor<3xsi32>
// CHECK-NOT: poptorch.max_dim
// CHECK: %[[SQUEEZED_IDXS:.*]] = "poptorch.view"(%[[IDXS]]) {shape = [3]} : (tensor<1x3xsi32>) -> tensor<3xsi32>
// CHECK-NOT: poptorch.max_dim
// CHECK: return %[[SQUEEZED_VALS]], %[[SQUEEZED_IDXS]] : tensor<3xsi32>, tensor<3xsi32>


// -----

// Test case: `max_dim(keep_dim=true)` should canon. to `topk(1)`.

func.func @max_dim_keepdim(%input: tensor<2x3xsi32>) -> (tensor<1x3xsi32>, tensor<1x3xsi32>) {
  %vals, %idxs = "poptorch.max_dim"(%input) {dim = 0, keepdim = true} : (tensor<2x3xsi32>) -> (tensor<1x3xsi32>, tensor<1x3xsi32>)
  return %vals, %idxs : tensor<1x3xsi32>, tensor<1x3xsi32>
}

// CHECK-LABEL: @max_dim_keepdim(
// CHECK-SAME:                   %[[INPUT:.*]]: tensor<2x3xsi32>
// CHECK-NOT: poptorch.max_dim
// CHECK: %[[VALS:.*]], %[[IDXS:.*]] = "poptorch.topk"(%[[INPUT]]) {
// CHECK-SAME:                                                      K = 1
// CHECK-SAME:                                                      dim = 0
// CHECK-SAME:                                                      largest = true
// CHECK-SAME:                                                      sorted = false
// CHECK-SAME:                                                     } : (tensor<2x3xsi32>) -> (tensor<1x3xsi32>, tensor<1x3xsi32>)
// CHECK-NOT: poptorch.max_dim
// CHECK: return %[[VALS]], %[[IDXS]] : tensor<1x3xsi32>, tensor<1x3xsi32>


// -----

// Test case: `min_dim` of a zero-sized tensor (ie. wrapped scalar) -> tuple
// (`clone(input)`, `zeros_like(result[0])`)
//
// (the `zeros_like` will then canonicalise to a `full`)

func.func @min_dim_scalar(%input: tensor<si32>) -> (tensor<si32>, tensor<1xsi32>) {
  %vals, %idxs = "poptorch.min_dim"(%input) {dim = 0, keepdim = false} : (tensor<si32>) -> (tensor<si32>, tensor<1xsi32>)

  // Write to the value-result, so that the `clone` doesn't get removed too.
  "poptorch.overwrite"(%vals, %input) : (tensor<si32>, tensor<si32>) -> ()

  return %vals, %idxs : tensor<si32>, tensor<1xsi32>
}

// CHECK-LABEL: @min_dim_scalar(
// CHECK-SAME:                  %[[INPUT:.*]]: tensor<si32>
// CHECK-NOT: poptorch.min_dim
// CHECK: %[[VALS:.*]] = "poptorch.clone"(%[[INPUT]]) : (tensor<si32>) -> tensor<si32>
// CHECK-NOT: poptorch.min_dim
// CHECK: %[[IDXS:.*]] = "poptorch.full"() {fill_value = 0
// CHECK-SAME:                                            } : () -> tensor<1xsi32>
// CHECK-NOT: poptorch.min_dim
// CHECK: return %[[VALS]], %[[IDXS]] : tensor<si32>, tensor<1xsi32>


// -----

// Test case: `min_dim(keep_dim=false)` should canon. to `topk(1)`, with the
// output values & indices then going through `squeeze_dim`.
//
// (the `squeeze_dim` will then canonicalise to a `view`)

func.func @min_dim_nokeepdim(%input: tensor<2x3xsi32>) -> (tensor<3xsi32>, tensor<3xsi32>) {
  %vals, %idxs = "poptorch.min_dim"(%input) {dim = 0, keepdim = false} : (tensor<2x3xsi32>) -> (tensor<3xsi32>, tensor<3xsi32>)
  return %vals, %idxs : tensor<3xsi32>, tensor<3xsi32>
}

// CHECK-LABEL: @min_dim_nokeepdim(
// CHECK-SAME:                     %[[INPUT:.*]]: tensor<2x3xsi32>
// CHECK-NOT: poptorch.min_dim
// CHECK: %[[VALS:.*]], %[[IDXS:.*]] = "poptorch.topk"(%[[INPUT]]) {
// CHECK-SAME:                                                      K = 1
// CHECK-SAME:                                                      dim = 0
// CHECK-SAME:                                                      largest = false
// CHECK-SAME:                                                      sorted = false
// CHECK-SAME:                                                     } : (tensor<2x3xsi32>) -> (tensor<1x3xsi32>, tensor<1x3xsi32>)
// CHECK-NOT: poptorch.min_dim
// CHECK: %[[SQUEEZED_VALS:.*]] = "poptorch.view"(%[[VALS]]) {shape = [3]} : (tensor<1x3xsi32>) -> tensor<3xsi32>
// CHECK-NOT: poptorch.min_dim
// CHECK: %[[SQUEEZED_IDXS:.*]] = "poptorch.view"(%[[IDXS]]) {shape = [3]} : (tensor<1x3xsi32>) -> tensor<3xsi32>
// CHECK-NOT: poptorch.min_dim
// CHECK: return %[[SQUEEZED_VALS]], %[[SQUEEZED_IDXS]] : tensor<3xsi32>, tensor<3xsi32>


// -----

// Test case: `min_dim(keep_dim=true)` should canon. to `topk(1)`.

func.func @min_dim_keepdim(%input: tensor<2x3xsi32>) -> (tensor<1x3xsi32>, tensor<1x3xsi32>) {
  %vals, %idxs = "poptorch.min_dim"(%input) {dim = 0, keepdim = true} : (tensor<2x3xsi32>) -> (tensor<1x3xsi32>, tensor<1x3xsi32>)
  return %vals, %idxs : tensor<1x3xsi32>, tensor<1x3xsi32>
}

// CHECK-LABEL: @min_dim_keepdim(
// CHECK-SAME:                   %[[INPUT:.*]]: tensor<2x3xsi32>
// CHECK-NOT: poptorch.min_dim
// CHECK: %[[VALS:.*]], %[[IDXS:.*]] = "poptorch.topk"(%[[INPUT]]) {
// CHECK-SAME:                                                      K = 1
// CHECK-SAME:                                                      dim = 0
// CHECK-SAME:                                                      largest = false
// CHECK-SAME:                                                      sorted = false
// CHECK-SAME:                                                     } : (tensor<2x3xsi32>) -> (tensor<1x3xsi32>, tensor<1x3xsi32>)
// CHECK-NOT: poptorch.min_dim
// CHECK: return %[[VALS]], %[[IDXS]] : tensor<1x3xsi32>, tensor<1x3xsi32>


// -----

// Test case: `median_dim_values` of a zero-sized tensor (ie. wrapped scalar) -> tuple
// (`clone(input)`, `zeros_like(result[0])`)
//
// (the `zeros_like` will then canonicalise to a `full`)

func.func @median_dim_values_scalar(%input: tensor<si32>) -> (tensor<si32>, tensor<1xsi32>) {
  %vals, %idxs = "poptorch.median_dim_values"(%input) {dim = 0, keepdim = false} : (tensor<si32>) -> (tensor<si32>, tensor<1xsi32>)

  // Write to the value-result, so that the `clone` doesn't get removed too.
  "poptorch.overwrite"(%vals, %input) : (tensor<si32>, tensor<si32>) -> ()

  return %vals, %idxs : tensor<si32>, tensor<1xsi32>
}

// CHECK-LABEL: @median_dim_values_scalar(
// CHECK-SAME:                  %[[INPUT:.*]]: tensor<si32>
// CHECK-NOT: poptorch.median_dim_values
// CHECK: %[[VALS:.*]] = "poptorch.clone"(%[[INPUT]]) : (tensor<si32>) -> tensor<si32>
// CHECK-NOT: poptorch.median_dim_values
// CHECK: %[[IDXS:.*]] = "poptorch.full"() {fill_value = 0
// CHECK-SAME:                                            } : () -> tensor<1xsi32>
// CHECK-NOT: poptorch.median_dim_values
// CHECK: return %[[VALS]], %[[IDXS]] : tensor<si32>, tensor<1xsi32>


// -----

// Test case: `median_dim_values(keepdim=false)` should canon. to a `topk`, with
// a `slice -> squeeze` for each return value.
//
// (the `squeeze_dim` will then canon. to a `view`)

func.func @median_dim_values_nokeepdim(%input: tensor<4x5xsi32>) -> (tensor<4xsi32>, tensor<4xsi32>) {
  %vals, %idxs = "poptorch.median_dim_values"(%input) {dim = 1, keepdim = false} : (tensor<4x5xsi32>) -> (tensor<4xsi32>, tensor<4xsi32>)
  return %vals, %idxs : tensor<4xsi32>, tensor<4xsi32>
}

// CHECK-LABEL: @median_dim_values_nokeepdim(
// CHECK-SAME:                               %[[INPUT:.*]]: tensor<4x5xsi32>
// CHECK-NOT: poptorch.median_dim_values
// CHECK: %[[VALS:.*]], %[[IDXS:.*]] = "poptorch.topk"(%[[INPUT]]) {
// CHECK-SAME:                                                      K = 3
// CHECK-SAME:                                                      dim = 1
// CHECK-SAME:                                                      largest = false
// CHECK-SAME:                                                      sorted = true
// CHECK-SAME:                                                     } : (tensor<4x5xsi32>) -> (tensor<4x3xsi32>, tensor<4x3xsi32>)
// CHECK-NOT: poptorch.median_dim_values
// CHECK: %[[SLICED_VALS:.*]] = "poptorch.slice_Tensor"(%[[VALS]]) {
// CHECK-SAME:                                                      dim = 1
// CHECK-SAME:                                                      end = 3
// CHECK-SAME:                                                      start = 2
// CHECK-SAME:                                                      step = 1
// CHECK-SAME:                                                     } : (tensor<4x3xsi32>) -> tensor<4x1xsi32>
// CHECK-NOT: poptorch.median_dim_values
// CHECK: %[[SLICED_IDXS:.*]] = "poptorch.slice_Tensor"(%[[IDXS]]) {
// CHECK-SAME:                                                      dim = 1
// CHECK-SAME:                                                      end = 3
// CHECK-SAME:                                                      start = 2
// CHECK-SAME:                                                      step = 1
// CHECK-SAME:                                                     } : (tensor<4x3xsi32>) -> tensor<4x1xsi32>
// CHECK-NOT: poptorch.median_dim_values
// CHECK: %[[SQUEEZED_VALS:.*]] = "poptorch.view"(%[[SLICED_VALS]]) {shape = [4]} : (tensor<4x1xsi32>) -> tensor<4xsi32>
// CHECK-NOT: poptorch.median_dim_values
// CHECK: %[[SQUEEZED_IDXS:.*]] = "poptorch.view"(%[[SLICED_IDXS]]) {shape = [4]} : (tensor<4x1xsi32>) -> tensor<4xsi32>
// CHECK-NOT: poptorch.median_dim_values
// CHECK: return %[[SQUEEZED_VALS]], %[[SQUEEZED_IDXS]] : tensor<4xsi32>, tensor<4xsi32>


// -----

// Test case: `median_dim_values(keepdim=true)` should canon. to a `topk`, with
// a `slice` for each return value.

func.func @median_dim_values_keepdim(%input: tensor<4x5xsi32>) -> (tensor<4x1xsi32>, tensor<4x1xsi32>) {
  %vals, %idxs = "poptorch.median_dim_values"(%input) {dim = 1, keepdim = true} : (tensor<4x5xsi32>) -> (tensor<4x1xsi32>, tensor<4x1xsi32>)
  return %vals, %idxs : tensor<4x1xsi32>, tensor<4x1xsi32>
}

// CHECK-LABEL: @median_dim_values_keepdim(
// CHECK-SAME:                             %[[INPUT:.*]]: tensor<4x5xsi32>
// CHECK-NOT: poptorch.median_dim_values
// CHECK: %[[VALS:.*]], %[[IDXS:.*]] = "poptorch.topk"(%[[INPUT]]) {
// CHECK-SAME:                                                      K = 3
// CHECK-SAME:                                                      dim = 1
// CHECK-SAME:                                                      largest = false
// CHECK-SAME:                                                      sorted = true
// CHECK-SAME:                                                     } : (tensor<4x5xsi32>) -> (tensor<4x3xsi32>, tensor<4x3xsi32>)
// CHECK-NOT: poptorch.median_dim_values
// CHECK: %[[SLICED_VALS:.*]] = "poptorch.slice_Tensor"(%[[VALS]]) {
// CHECK-SAME:                                                      dim = 1
// CHECK-SAME:                                                      end = 3
// CHECK-SAME:                                                      start = 2
// CHECK-SAME:                                                      step = 1
// CHECK-SAME:                                                     } : (tensor<4x3xsi32>) -> tensor<4x1xsi32>
// CHECK-NOT: poptorch.median_dim_values
// CHECK: %[[SLICED_IDXS:.*]] = "poptorch.slice_Tensor"(%[[IDXS]]) {
// CHECK-SAME:                                                      dim = 1
// CHECK-SAME:                                                      end = 3
// CHECK-SAME:                                                      start = 2
// CHECK-SAME:                                                      step = 1
// CHECK-SAME:                                                     } : (tensor<4x3xsi32>) -> tensor<4x1xsi32>
// CHECK-NOT: poptorch.median_dim_values
// CHECK: return %[[SLICED_VALS]], %[[SLICED_IDXS]] : tensor<4x1xsi32>, tensor<4x1xsi32>


// -----

// Test case: `median_dim_values(keepdim=true)`, with a negative dim.

func.func @median_dim_values_negative_dim(%input: tensor<4x5xsi32>) -> (tensor<4x1xsi32>, tensor<4x1xsi32>) {
  %vals, %idxs = "poptorch.median_dim_values"(%input) {dim = -1, keepdim = true} : (tensor<4x5xsi32>) -> (tensor<4x1xsi32>, tensor<4x1xsi32>)
  return %vals, %idxs : tensor<4x1xsi32>, tensor<4x1xsi32>
}

// CHECK-LABEL: @median_dim_values_negative_dim(
// CHECK-SAME:                                  %[[INPUT:.*]]: tensor<4x5xsi32>
// CHECK-NOT: poptorch.median_dim_values
// CHECK: %[[VALS:.*]], %[[IDXS:.*]] = "poptorch.topk"(%[[INPUT]]) {
// CHECK-SAME:                                                      K = 3
// CHECK-SAME:                                                      dim = 1
// CHECK-SAME:                                                      largest = false
// CHECK-SAME:                                                      sorted = true
// CHECK-SAME:                                                     } : (tensor<4x5xsi32>) -> (tensor<4x3xsi32>, tensor<4x3xsi32>)
// CHECK-NOT: poptorch.median_dim_values
// CHECK: %[[SLICED_VALS:.*]] = "poptorch.slice_Tensor"(%[[VALS]]) {
// CHECK-SAME:                                                      dim = 1
// CHECK-SAME:                                                      end = 3
// CHECK-SAME:                                                      start = 2
// CHECK-SAME:                                                      step = 1
// CHECK-SAME:                                                     } : (tensor<4x3xsi32>) -> tensor<4x1xsi32>
// CHECK-NOT: poptorch.median_dim_values
// CHECK: %[[SLICED_IDXS:.*]] = "poptorch.slice_Tensor"(%[[IDXS]]) {
// CHECK-SAME:                                                      dim = 1
// CHECK-SAME:                                                      end = 3
// CHECK-SAME:                                                      start = 2
// CHECK-SAME:                                                      step = 1
// CHECK-SAME:                                                     } : (tensor<4x3xsi32>) -> tensor<4x1xsi32>
// CHECK-NOT: poptorch.median_dim_values
// CHECK: return %[[SLICED_VALS]], %[[SLICED_IDXS]] : tensor<4x1xsi32>, tensor<4x1xsi32>
