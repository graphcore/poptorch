// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// RUN: poptorch-opt --canonicalize %s --split-input-file | FileCheck %s


// Test case: `add(bool)` -> `logicalOr`

func.func @add_bool(%input: tensor<3xui1>, %other: tensor<3xui1>) -> tensor<3xui1> {
  %res = "poptorch.add"(%input, %other) : (tensor<3xui1>, tensor<3xui1>) -> tensor<3xui1>
  return %res : tensor<3xui1>
}

// CHECK-LABEL: @add_bool(
// CHECK-SAME:            %[[INPUT:.*]]: tensor<3xui1>, %[[OTHER:.*]]: tensor<3xui1>
// CHECK-NOT: poptorch.add
// CHECK: %[[RES:.*]] = poptorch.logicalOr(%[[INPUT]], %[[OTHER]]) (tensor<3xui1>, tensor<3xui1>) -> tensor<3xui1>
// CHECK-NOT: poptorch.add
// CHECK: return %[[RES]] : tensor<3xui1>


// -----

// Test case: `maximum(bool)` -> `logicalOr`

func.func @maximum_bool(%input: tensor<3xui1>, %other: tensor<3xui1>) -> tensor<3xui1> {
  %res = "poptorch.maximum"(%input, %other) : (tensor<3xui1>, tensor<3xui1>) -> tensor<3xui1>
  return %res : tensor<3xui1>
}

// CHECK-LABEL: @maximum_bool(
// CHECK-SAME:                %[[INPUT:.*]]: tensor<3xui1>, %[[OTHER:.*]]: tensor<3xui1>
// CHECK-NOT: poptorch.maximum
// CHECK: %[[RES:.*]] = poptorch.logicalOr(%[[INPUT]], %[[OTHER]]) (tensor<3xui1>, tensor<3xui1>) -> tensor<3xui1>
// CHECK-NOT: poptorch.maximum
// CHECK: return %[[RES]] : tensor<3xui1>


// -----

// Test case: `mul(bool)` -> `logicalAnd`

func.func @mul_bool(%input: tensor<3xui1>, %other: tensor<3xui1>) -> tensor<3xui1> {
  %res = "poptorch.mul"(%input, %other) : (tensor<3xui1>, tensor<3xui1>) -> tensor<3xui1>
  return %res : tensor<3xui1>
}

// CHECK-LABEL: @mul_bool(
// CHECK-SAME:            %[[INPUT:.*]]: tensor<3xui1>, %[[OTHER:.*]]: tensor<3xui1>
// CHECK-NOT: poptorch.mul
// CHECK: %[[RES:.*]] = poptorch.logicalAnd(%[[INPUT]], %[[OTHER]]) (tensor<3xui1>, tensor<3xui1>) -> tensor<3xui1>
// CHECK-NOT: poptorch.mul
// CHECK: return %[[RES]] : tensor<3xui1>


// -----

// Test case: `minimum(bool)` -> `logicalAnd`

func.func @minimum_bool(%input: tensor<3xui1>, %other: tensor<3xui1>) -> tensor<3xui1> {
  %res = "poptorch.minimum"(%input, %other) : (tensor<3xui1>, tensor<3xui1>) -> tensor<3xui1>
  return %res : tensor<3xui1>
}

// CHECK-LABEL: @minimum_bool(
// CHECK-SAME:                %[[INPUT:.*]]: tensor<3xui1>, %[[OTHER:.*]]: tensor<3xui1>
// CHECK-NOT: poptorch.minimum
// CHECK: %[[RES:.*]] = poptorch.logicalAnd(%[[INPUT]], %[[OTHER]]) (tensor<3xui1>, tensor<3xui1>) -> tensor<3xui1>
// CHECK-NOT: poptorch.minimum
// CHECK: return %[[RES]] : tensor<3xui1>


// -----

// Test case: `bitwiseAnd(bool)` -> `logicalAnd`

func.func @bitwiseAnd_bool(%input: tensor<3xui1>, %other: tensor<3xui1>) -> tensor<3xui1> {
  %res = "poptorch.bitwiseAnd"(%input, %other) : (tensor<3xui1>, tensor<3xui1>) -> tensor<3xui1>
  return %res : tensor<3xui1>
}

// CHECK-LABEL: @bitwiseAnd_bool(
// CHECK-SAME:                   %[[INPUT:.*]]: tensor<3xui1>, %[[OTHER:.*]]: tensor<3xui1>
// CHECK-NOT: poptorch.bitwiseAnd
// CHECK: %[[RES:.*]] = poptorch.logicalAnd(%[[INPUT]], %[[OTHER]]) (tensor<3xui1>, tensor<3xui1>) -> tensor<3xui1>
// CHECK-NOT: poptorch.bitwiseAnd
// CHECK: return %[[RES]] : tensor<3xui1>


// -----

// Test case: `bitwiseOr(bool)` -> `logicalOr`

func.func @bitwiseOr_bool(%input: tensor<3xui1>, %other: tensor<3xui1>) -> tensor<3xui1> {
  %res = "poptorch.bitwiseOr"(%input, %other) : (tensor<3xui1>, tensor<3xui1>) -> tensor<3xui1>
  return %res : tensor<3xui1>
}

// CHECK-LABEL: @bitwiseOr_bool(
// CHECK-SAME:                  %[[INPUT:.*]]: tensor<3xui1>, %[[OTHER:.*]]: tensor<3xui1>
// CHECK-NOT: poptorch.bitwiseOr
// CHECK: %[[RES:.*]] = poptorch.logicalOr(%[[INPUT]], %[[OTHER]]) (tensor<3xui1>, tensor<3xui1>) -> tensor<3xui1>
// CHECK-NOT: poptorch.bitwiseOr
// CHECK: return %[[RES]] : tensor<3xui1>


// -----

// Test case: `bitwiseXor(bool)` -> `neq`

func.func @bitwiseXor_bool(%input: tensor<3xui1>, %other: tensor<3xui1>) -> tensor<3xui1> {
  %res = "poptorch.bitwiseXor"(%input, %other) : (tensor<3xui1>, tensor<3xui1>) -> tensor<3xui1>
  return %res : tensor<3xui1>
}

// CHECK-LABEL: @bitwiseXor_bool(
// CHECK-SAME:                  %[[INPUT:.*]]: tensor<3xui1>, %[[OTHER:.*]]: tensor<3xui1>
// CHECK-NOT: poptorch.bitwiseXor
// CHECK: %[[RES:.*]] = poptorch.neq(%[[INPUT]], %[[OTHER]]) (tensor<3xui1>, tensor<3xui1>) -> tensor<3xui1>
// CHECK-NOT: poptorch.bitwiseXor
// CHECK: return %[[RES]] : tensor<3xui1>


// -----

// Test case: `bitwiseXnor(bool)` -> `eq`

func.func @bitwiseXnor_bool(%input: tensor<3xui1>, %other: tensor<3xui1>) -> tensor<3xui1> {
  %res = "poptorch.bitwiseXnor"(%input, %other) : (tensor<3xui1>, tensor<3xui1>) -> tensor<3xui1>
  return %res : tensor<3xui1>
}

// CHECK-LABEL: @bitwiseXnor_bool(
// CHECK-SAME:                    %[[INPUT:.*]]: tensor<3xui1>, %[[OTHER:.*]]: tensor<3xui1>
// CHECK-NOT: poptorch.bitwiseXnor
// CHECK: %[[RES:.*]] = poptorch.eq(%[[INPUT]], %[[OTHER]]) (tensor<3xui1>, tensor<3xui1>) -> tensor<3xui1>
// CHECK-NOT: poptorch.bitwiseXnor
// CHECK: return %[[RES]] : tensor<3xui1>


// -----

// Test case: `bitwiseNot(bool)` -> `logicalNot`

func.func @bitwiseNot_bool(%input: tensor<3xui1>) -> tensor<3xui1> {
  %res = "poptorch.bitwiseNot"(%input) : (tensor<3xui1>) -> tensor<3xui1>
  return %res : tensor<3xui1>
}

// CHECK-LABEL: @bitwiseNot_bool(
// CHECK-SAME:                   %[[INPUT:.*]]: tensor<3xui1>
// CHECK-NOT: poptorch.bitwiseNot
// CHECK: %[[RES:.*]] = poptorch.logicalNot(%[[INPUT]]) (tensor<3xui1>) -> tensor<3xui1>
// CHECK-NOT: poptorch.bitwiseNot
// CHECK: return %[[RES]] : tensor<3xui1>


// -----

// Test case: `isnan(int)` canons. to (`zeros_like` canons. to `full`)

func.func @isnan_int(%input: tensor<3xsi32>) -> tensor<3xui1> {
  %res = "poptorch.isnan"(%input) : (tensor<3xsi32>) -> tensor<3xui1>
  return %res : tensor<3xui1>
}

// CHECK-LABEL: @isnan_int(
// CHECK-SAME:             %[[INPUT:.*]]: tensor<3xsi32>
// CHECK-NOT: poptorch.isnan
// CHECK: %[[RES:.*]] = "poptorch.full"() {fill_value = 0
// CHECK-SAME:                                           } : () -> tensor<3xui1>
// CHECK-NOT: poptorch.isnan
// CHECK: return %[[RES]] : tensor<3xui1>


// -----

// Test case: `signum(bool)` -> `clone`

func.func @signum_bool(%input: tensor<3xui1>) -> tensor<3xui1> {
  %res = "poptorch.signum"(%input) : (tensor<3xui1>) -> tensor<3xui1>
  // Write to the result, so that the `clone` doesn't get removed too.
  "poptorch.overwrite"(%res, %input) : (tensor<3xui1>, tensor<3xui1>) -> ()
  return %res : tensor<3xui1>
}

// CHECK-LABEL: @signum_bool(
// CHECK-SAME:               %[[INPUT:.*]]: tensor<3xui1>
// CHECK-NOT: poptorch.signum
// CHECK: %[[RES:.*]] = "poptorch.clone"(%[[INPUT]])
// CHECK-NOT: poptorch.signum
// CHECK: return %[[RES]] : tensor<3xui1>
