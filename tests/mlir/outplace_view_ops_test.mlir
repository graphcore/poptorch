// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// RUN: poptorch-opt --outplace-view-ops %s --split-input-file | FileCheck %s

// Test case: simple case of outplacing a view op.

// CHECK: @f
// CHECK: poptorch.copy_from_host
// CHECK-NOT: poptorch.view
// CHECK: poptorch.viewOutplace
// CHECK-NOT: poptorch.view
// CHECK: poptorch.copy_to_host

func.func @f() {
  %0 = "poptorch.copy_from_host"() {handle = "Input/0"} : () -> tensor<3xsi32>
  %1 = "poptorch.view"(%0) {shape = [3, 1]} : (tensor<3xsi32>) -> tensor<3x1xsi32>
  "poptorch.copy_to_host"(%1) {handle = "Output/0"} : (tensor<3x1xsi32>) -> ()
  return
}

// -----

// Test case: simple view, with an inplace op after.

// CHECK: @f
// CHECK: %[[VIEWED:.*]] = "poptorch.viewOutplace"(%[[INPUT:.*]]) {{{.*}}
// CHECK: %[[ADDED:.*]] = poptorch.add(%[[VIEWED]],
// CHECK: %[[VIEW_INV:.*]] = "poptorch.viewInverse"(%[[ADDED]], %[[INPUT]])
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[VIEW_INV]]
// CHECK: poptorch.copy_to_host(%[[INPUT]]

func.func @f() {
  %0 = "poptorch.copy_from_host"() {handle = "Input/0"} : () -> tensor<3xsi32>
  %1 = "poptorch.view"(%0) {shape = [3, 1]} : (tensor<3xsi32>) -> tensor<3x1xsi32>
  %2 = "poptorch.tensorconstant_int"() {data = [5 : i32], shape = []} : () -> tensor<si32>
  %3 = "poptorch.add"(%1, %2) {alpha = 1.000000e+00 : f32} : (tensor<3x1xsi32>, tensor<si32>) -> tensor<3x1xsi32>
  "poptorch.overwrite"(%1, %3) : (tensor<3x1xsi32>, tensor<3x1xsi32>) -> ()
  "poptorch.copy_to_host"(%0) {handle = "Output/0"} : (tensor<3xsi32>) -> ()
  return
}
