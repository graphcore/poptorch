// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// RUN: poptorch-opt --canonicalize %s --split-input-file | FileCheck %s


// Test case: `squeeze` -> `view`

func.func @squeeze(%input: tensor<3x1xsi32>) -> tensor<3xsi32> {
  %res = "poptorch.squeeze"(%input) : (tensor<3x1xsi32>) -> tensor<3xsi32>
  return %res : tensor<3xsi32>
}

// CHECK-LABEL: @squeeze(
// CHECK-SAME:           %[[INPUT:.*]]: tensor<3x1xsi32>
// CHECK-NOT: poptorch.squeeze
// CHECK: %[[RES:.*]] = "poptorch.view"(%[[INPUT]]) {shape = [3]} : (tensor<3x1xsi32>) -> tensor<3xsi32>
// CHECK-NOT: poptorch.squeeze
// CHECK: return %[[RES]] : tensor<3xsi32>


// -----

// Test case: `squeeze_dim` -> `view`

func.func @squeeze_dim(%input: tensor<3x1xsi32>) -> tensor<3xsi32> {
  %res = "poptorch.squeeze_dim"(%input) {dim = 0} : (tensor<3x1xsi32>) -> tensor<3xsi32>
  return %res : tensor<3xsi32>
}

// CHECK-LABEL: @squeeze_dim(
// CHECK-SAME:               %[[INPUT:.*]]: tensor<3x1xsi32>
// CHECK-NOT: poptorch.squeeze_dim
// CHECK: %[[RES:.*]] = "poptorch.view"(%[[INPUT]]) {shape = [3]} : (tensor<3x1xsi32>) -> tensor<3xsi32>
// CHECK-NOT: poptorch.squeeze_dim
// CHECK: return %[[RES]] : tensor<3xsi32>


// -----

// Test case: `unsqueeze` -> `view`

func.func @unsqueeze(%input: tensor<3xsi32>) -> tensor<1x3xsi32> {
  %res = "poptorch.unsqueeze"(%input) {dim = 0} : (tensor<3xsi32>) -> tensor<1x3xsi32>
  return %res : tensor<1x3xsi32>
}

// CHECK-LABEL: @unsqueeze(
// CHECK-SAME:           %[[INPUT:.*]]: tensor<3xsi32>
// CHECK-NOT: poptorch.unsqueeze
// CHECK: %[[RES:.*]] = "poptorch.view"(%[[INPUT]]) {shape = [1, 3]} : (tensor<3xsi32>) -> tensor<1x3xsi32>
// CHECK-NOT: poptorch.unsqueeze
// CHECK: return %[[RES]] : tensor<1x3xsi32>


// -----

// Test case: `as_strided` -> `view`

func.func @as_strided(%input: tensor<3xsi32>) -> tensor<1x3xsi32> {
  %res = "poptorch.as_strided"(%input) {size = [1, 3], strides = [3, 1]} : (tensor<3xsi32>) -> tensor<1x3xsi32>
  return %res : tensor<1x3xsi32>
}

// CHECK-LABEL: @as_strided(
// CHECK-SAME:              %[[INPUT:.*]]: tensor<3xsi32>
// CHECK-NOT: poptorch.as_strided
// CHECK: %[[RES:.*]] = "poptorch.view"(%[[INPUT]]) {shape = [1, 3]} : (tensor<3xsi32>) -> tensor<1x3xsi32>
// CHECK-NOT: poptorch.as_strided
// CHECK: return %[[RES]] : tensor<1x3xsi32>


// -----

// Test case: `alias` -> `view`

func.func @alias(%input: tensor<3xsi32>) -> tensor<3xsi32> {
  %res = "poptorch.alias"(%input) : (tensor<3xsi32>) -> tensor<3xsi32>
  return %res : tensor<3xsi32>
}

// CHECK-LABEL: @alias(
// CHECK-SAME:         %[[INPUT:.*]]: tensor<3xsi32>
// CHECK-NOT: poptorch.alias
// CHECK: %[[RES:.*]] = "poptorch.view"(%[[INPUT]]) {shape = [3]} : (tensor<3xsi32>) -> tensor<3xsi32>
// CHECK-NOT: poptorch.alias
// CHECK: return %[[RES]] : tensor<3xsi32>


// -----

// Test case: `detach` -> `view`

func.func @detach(%input: tensor<3xsi32>) -> tensor<3xsi32> {
  %res = "poptorch.detach"(%input) : (tensor<3xsi32>) -> tensor<3xsi32>
  return %res : tensor<3xsi32>
}

// CHECK-LABEL: @detach(
// CHECK-SAME:          %[[INPUT:.*]]: tensor<3xsi32>
// CHECK-NOT: poptorch.detach
// CHECK: %[[RES:.*]] = "poptorch.view"(%[[INPUT]]) {shape = [3]} : (tensor<3xsi32>) -> tensor<3xsi32>
// CHECK-NOT: poptorch.detach
// CHECK: return %[[RES]] : tensor<3xsi32>


// -----

// Test case: `transpose` -> `permute`

func.func @transpose(%input: tensor<1x2x3xsi32>) -> tensor<1x3x2xsi32> {
  %res = "poptorch.transpose"(%input) {dim0 = 1, dim1 = 2} : (tensor<1x2x3xsi32>) -> tensor<1x3x2xsi32>
  return %res : tensor<1x3x2xsi32>
}

// CHECK-LABEL: @transpose(
// CHECK-SAME:             %[[INPUT:.*]]: tensor<1x2x3xsi32>
// CHECK-NOT: poptorch.transpose
// CHECK: %[[RES:.*]] = "poptorch.permute"(%[[INPUT]]) {dims = [0, 2, 1]} : (tensor<1x2x3xsi32>) -> tensor<1x3x2xsi32>
// CHECK-NOT: poptorch.transpose
// CHECK: return %[[RES]] : tensor<1x3x2xsi32>


// -----

// Test case: `permuteInverse` -> `permuteOutplace`

func.func @permuteInverse(%input: tensor<3x1x2xsi32>, %original: tensor<1x2x3xsi32>) -> tensor<1x2x3xsi32> {
  %res = "poptorch.permuteInverse"(%input, %original) {dims = [2, 0, 1]} : (tensor<3x1x2xsi32>, tensor<1x2x3xsi32>) -> tensor<1x2x3xsi32>
  return %res : tensor<1x2x3xsi32>
}

// CHECK-LABEL: @permuteInverse(
// CHECK-SAME:                  %[[INPUT:.*]]: tensor<3x1x2xsi32>
// CHECK-NOT: poptorch.permuteInverse
// CHECK: %[[RES:.*]] = "poptorch.permuteOutplace"(%[[INPUT]]) {dims = [1, 2, 0]} : (tensor<3x1x2xsi32>) -> tensor<1x2x3xsi32>
// CHECK-NOT: poptorch.permuteInverse
// CHECK: return %[[RES]] : tensor<1x2x3xsi32>


// -----

// Test case: `viewInverse` -> `viewOutplace`

func.func @viewInverse(%input: tensor<3x1x2xsi32>, %original: tensor<1x2x3xsi32>) -> tensor<1x2x3xsi32> {
  %res = "poptorch.viewInverse"(%input, %original) {shape = [3, 1, 2]} : (tensor<3x1x2xsi32>, tensor<1x2x3xsi32>) -> tensor<1x2x3xsi32>
  return %res : tensor<1x2x3xsi32>
}

// CHECK-LABEL: @viewInverse(
// CHECK-SAME:               %[[INPUT:.*]]: tensor<3x1x2xsi32>
// CHECK-NOT: poptorch.viewInverse
// CHECK: %[[RES:.*]] = "poptorch.viewOutplace"(%[[INPUT]]) {shape = [1, 2, 3]} : (tensor<3x1x2xsi32>) -> tensor<1x2x3xsi32>
// CHECK-NOT: poptorch.viewInverse
// CHECK: return %[[RES]] : tensor<1x2x3xsi32>


// -----

// Test case: `select` canons. to (`slice.Tensor` -> (`squeeze_dim` canons. to `view`))

func.func @select(%input: tensor<1x2x3xsi32>) -> tensor<1x2xsi32> {
  %res = "poptorch.select"(%input) {dim = 2, idx = 1} : (tensor<1x2x3xsi32>) -> tensor<1x2xsi32>
  return %res : tensor<1x2xsi32>
}

// CHECK-LABEL: @select(
// CHECK-SAME:          %[[INPUT:.*]]: tensor<1x2x3xsi32>
// CHECK-NOT: poptorch.select
// CHECK: %[[SLICED:.*]] = "poptorch.slice_Tensor"(%[[INPUT]]) {
// CHECK-SAME:                                                  dim = 2
// CHECK-SAME:                                                  end = 2
// CHECK-SAME:                                                  start = 1
// CHECK-SAME:                                                  step = 1
// CHECK-SAME:                                                 } : (tensor<1x2x3xsi32>) -> tensor<1x2x1xsi32>
// CHECK-NOT: poptorch.select
// CHECK: %[[RES:.*]] = "poptorch.view"(%[[SLICED]]) {shape = [1, 2]} : (tensor<1x2x1xsi32>) -> tensor<1x2xsi32>
// CHECK-NOT: poptorch.select
// CHECK: return %[[RES]] : tensor<1x2xsi32>


// -----

// Test case: `viewOutplace` with identical input & output shapes should be
// removed.

func.func @viewOutplace(%input: tensor<1x2x3xsi32>) -> tensor<1x2x3xsi32> {
  %res = "poptorch.viewOutplace"(%input) {shape = [1, 2, 3]} : (tensor<1x2x3xsi32>) -> tensor<1x2x3xsi32>
  return %res : tensor<1x2x3xsi32>
}

// CHECK-LABEL: @viewOutplace(
// CHECK-SAME:               %[[INPUT:.*]]: tensor<1x2x3xsi32>
// CHECK-NOT: poptorch.viewOutplace
// CHECK: return %[[INPUT]] : tensor<1x2x3xsi32>
