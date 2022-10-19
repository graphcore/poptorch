// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// RUN: poptorch-opt --remove-overwrite %s --split-input-file | FileCheck %s

// Test case: simple case of deleting an overwrite.

// CHECK: @f
// CHECK-NOT: poptorch.overwrite
// CHECK: %[[ADDED:.*]] = poptorch.add
// CHECK-NOT: poptorch.overwrite
// CHECK: poptorch.copy_to_host(%[[ADDED]]
// CHECK-NOT: poptorch.overwrite

func.func @f() {
  %0 = "poptorch.copy_from_host"() {handle = "Input/0"} : () -> tensor<3xsi32>
  %1 = "poptorch.tensorconstant_int"() {data = [5 : i32], shape = []} : () -> tensor<si32>
  %2 = "poptorch.add"(%0, %1) {alpha = 1.000000e+00 : f32} : (tensor<3xsi32>, tensor<si32>) -> tensor<3xsi32>
  "poptorch.overwrite"(%0, %2) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  "poptorch.copy_to_host"(%0) {handle = "Output/0"} : (tensor<3xsi32>) -> ()
  return
}

// -----

// Test case: overwrite is applied to view of input.

// CHECK: @f
// CHECK: %[[VIEW_INV:.*]] = "poptorch.viewInverse"
// CHECK-NOT: poptorch.overwrite
// CHECK: "poptorch.viewOutplace"(%[[VIEW_INV]]

func.func @f() {
  %0 = "poptorch.copy_from_host"() {handle = "Input/0"} : () -> tensor<3xsi32>
  %1 = "poptorch.tensorconstant_int"() {data = [5 : i32], shape = []} : () -> tensor<si32>
  %2 = "poptorch.viewOutplace"(%0) {shape = [1, 3]} : (tensor<3xsi32>) -> tensor<1x3xsi32>
  %3 = "poptorch.add"(%2, %1) {alpha = 1.000000e+00 : f32} : (tensor<1x3xsi32>, tensor<si32>) -> tensor<1x3xsi32>
  %4 = "poptorch.viewInverse"(%3, %0) {shape = [1, 3]} : (tensor<1x3xsi32>, tensor<3xsi32>) -> tensor<3xsi32>
  "poptorch.overwrite"(%0, %4) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  %5 = "poptorch.viewOutplace"(%0) {shape = [1, 3]} : (tensor<3xsi32>) -> tensor<1x3xsi32>
  "poptorch.copy_to_host"(%5) {handle = "Output/0"} : (tensor<1x3xsi32>) -> ()
  return
}

