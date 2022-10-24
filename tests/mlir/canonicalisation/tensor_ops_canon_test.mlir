// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// RUN: poptorch-opt --canonicalize %s --split-input-file -allow-unregistered-dialect | FileCheck %s


// Test case: `ones` -> `full(1)`

func.func @ones() -> tensor<4x5xf32> {
  %res = "poptorch.ones"() {size = [4, 5]} : () -> tensor<4x5xf32>
  return %res : tensor<4x5xf32>
}

// CHECK-LABEL: @ones(
// CHECK-NOT: poptorch.ones
// CHECK: %[[RES:.*]] = "poptorch.full"() {fill_value = 1
// CHECK-SAME:                                           } : () -> tensor<4x5xf32>
// CHECK-NOT: poptorch.ones
// CHECK: return %[[RES]] : tensor<4x5xf32>


// -----

// Test case: `zeros_like` -> `full(0)`

func.func @zeros_like(%input: tensor<4x5xf32>) -> tensor<4x5xf32> {
  %res = "poptorch.zeros_like"(%input) : (tensor<4x5xf32>) -> tensor<4x5xf32>
  return %res : tensor<4x5xf32>
}

// CHECK-LABEL: @zeros_like(
// CHECK-SAME:              %[[INPUT:.*]]: tensor<4x5xf32>
// CHECK-NOT: poptorch.zeros_like
// CHECK: %[[RES:.*]] = "poptorch.full"() {fill_value = 0
// CHECK-SAME:                                           } : () -> tensor<4x5xf32>
// CHECK-NOT: poptorch.zeros_like
// CHECK: return %[[RES]] : tensor<4x5xf32>


// -----

// Test case: `full_like` -> `full(0)`

func.func @full_like(%input: tensor<4x5xf32>) -> tensor<4x5xf32> {
  %res = "poptorch.full_like"(%input) {fill_value = 42.5 : f32} : (tensor<4x5xf32>) -> tensor<4x5xf32>
  return %res : tensor<4x5xf32>
}

// CHECK-LABEL: @full_like(
// CHECK-SAME:              %[[INPUT:.*]]: tensor<4x5xf32>
// CHECK-NOT: poptorch.full_like
// CHECK: %[[RES:.*]] = "poptorch.full"() {fill_value = 4.25{{0*}}e+01 : f32} : () -> tensor<4x5xf32>
// CHECK-NOT: poptorch.full_like
// CHECK: return %[[RES]] : tensor<4x5xf32>


// -----

// Test case: `clone`: if neither source nor destination is written to
// afterwards, just swap out the IDs.

func.func @clone_neither_written_to(%input: tensor<4x5xf32>) -> tensor<4x5xf32> {
  %clone = "poptorch.clone"(%input) : (tensor<4x5xf32>) -> tensor<4x5xf32>
  return %clone : tensor<4x5xf32>
}

// CHECK-LABEL: @clone_neither_written_to(
// CHECK-SAME:                            %[[INPUT:.*]]: tensor<4x5xf32>
// CHECK-NOT: poptorch.clone
// CHECK: return %[[INPUT]] : tensor<4x5xf32>


// -----

// Test case: `clone`: write to the source, so that the `clone` *shouldn't* be
// removed.

func.func @clone_source_written_to(%input: tensor<4x5xf32>) -> tensor<4x5xf32> {
  %clone = "poptorch.clone"(%input) : (tensor<4x5xf32>) -> tensor<4x5xf32>
  "poptorch.overwrite"(%input, %clone) : (tensor<4x5xf32>, tensor<4x5xf32>) -> ()

  return %clone : tensor<4x5xf32>
}

// CHECK-LABEL: @clone_source_written_to(
// CHECK-SAME:                           %[[INPUT:.*]]: tensor<4x5xf32>
// CHECK: %[[CLONE:.*]] = "poptorch.clone"
// CHECK: poptorch.overwrite
// CHECK: return %[[CLONE]] : tensor<4x5xf32>


// -----

// Test case: `clone`: write to the destination, so that the `clone`
// *shouldn't* be removed.

func.func @clone_source_written_to(%input: tensor<4x5xf32>) -> tensor<4x5xf32> {
  %clone = "poptorch.clone"(%input) : (tensor<4x5xf32>) -> tensor<4x5xf32>
  "poptorch.overwrite"(%clone, %input) : (tensor<4x5xf32>, tensor<4x5xf32>) -> ()

  return %clone : tensor<4x5xf32>
}

// CHECK-LABEL: @clone_source_written_to(
// CHECK-SAME:                           %[[INPUT:.*]]: tensor<4x5xf32>
// CHECK: %[[CLONE:.*]] = "poptorch.clone"
// CHECK: poptorch.overwrite
// CHECK: return %[[CLONE]] : tensor<4x5xf32>


// -----

// Test case: `clone`: write to the source & destination, so that the `clone`
// *shouldn't* be removed.

func.func @clone_source_written_to(%input: tensor<4x5xf32>) -> tensor<4x5xf32> {
  %clone = "poptorch.clone"(%input) : (tensor<4x5xf32>) -> tensor<4x5xf32>
  "poptorch.overwrite"(%input, %clone) : (tensor<4x5xf32>, tensor<4x5xf32>) -> ()
  "poptorch.overwrite"(%clone, %input) : (tensor<4x5xf32>, tensor<4x5xf32>) -> ()

  return %clone : tensor<4x5xf32>
}

// CHECK-LABEL: @clone_source_written_to(
// CHECK-SAME:                           %[[INPUT:.*]]: tensor<4x5xf32>
// CHECK: %[[CLONE:.*]] = "poptorch.clone"
// CHECK: poptorch.overwrite
// CHECK: poptorch.overwrite
// CHECK: return %[[CLONE]] : tensor<4x5xf32>


// -----

// Test case: `clone`: write to a view of the source, so that the `clone`
// *shouldn't* be removed.

func.func @clone_source_view_written_to(%input: tensor<4x5xf32>, %other: tensor<5x4xf32>) -> tensor<4x5xf32> {
  %clone = "poptorch.clone"(%input) : (tensor<4x5xf32>) -> tensor<4x5xf32>

  %view = "poptorch.view"(%input) {shape = [5, 4]} : (tensor<4x5xf32>) -> tensor<5x4xf32>
  "poptorch.overwrite"(%view, %other) : (tensor<5x4xf32>, tensor<5x4xf32>) -> ()

  return %clone : tensor<4x5xf32>
}

// CHECK-LABEL: @clone_source_view_written_to(
// CHECK-SAME:                                %[[INPUT:.*]]: tensor<4x5xf32>
// CHECK: %[[CLONE:.*]] = "poptorch.clone"
// CHECK: poptorch.view
// CHECK: poptorch.overwrite
// CHECK: return %[[CLONE]] : tensor<4x5xf32>


// -----

// Test case: `clone`: write to a view of the source, so that the `clone`
// *shouldn't* be removed.

func.func @clone_dest_view_written_to(%input: tensor<4x5xf32>, %other: tensor<5x4xf32>) -> (tensor<4x5xf32>, tensor<4x5xf32>) {
  %clone = "poptorch.clone"(%input) : (tensor<4x5xf32>) -> tensor<4x5xf32>

  %view = "poptorch.view"(%clone) {shape = [5, 4]} : (tensor<4x5xf32>) -> tensor<5x4xf32>
  "poptorch.overwrite"(%view, %other) : (tensor<5x4xf32>, tensor<5x4xf32>) -> ()

  return %input, %clone : tensor<4x5xf32>, tensor<4x5xf32>
}

// CHECK-LABEL: @clone_dest_view_written_to(
// CHECK-SAME:                              %[[INPUT:.*]]: tensor<4x5xf32>
// CHECK: %[[CLONE:.*]] = "poptorch.clone"
// CHECK: poptorch.view
// CHECK: poptorch.overwrite
// CHECK: return %[[INPUT]], %[[CLONE]] : tensor<4x5xf32>, tensor<4x5xf32>


// -----

// Test case: `clone`: source is a view, so that the `clone` *shouldn't* be
// removed.

func.func @clone_of_view(%input: tensor<4x5xf32>, %other: tensor<5x4xf32>) -> tensor<5x4xf32> {
  %view = "poptorch.view"(%input) {shape = [5, 4]} : (tensor<4x5xf32>) -> tensor<5x4xf32>
  %clone = "poptorch.clone"(%view) : (tensor<5x4xf32>) -> tensor<5x4xf32>

  return %clone : tensor<5x4xf32>
}

// CHECK-LABEL: @clone_of_view(
// CHECK-SAME:                 %[[INPUT:.*]]: tensor<4x5xf32>
// CHECK: poptorch.view
// CHECK: %[[CLONE:.*]] = "poptorch.clone"
// CHECK: return %[[CLONE]] : tensor<5x4xf32>


// -----

// Test case: `clone`: write to the source *before* the `clone` is taken, so
// that it can still be removed.

func.func @clone_source_prewritten(%input: tensor<4x5xf32>, %other: tensor<4x5xf32>) -> tensor<4x5xf32> {
  "poptorch.overwrite"(%input, %other) : (tensor<4x5xf32>, tensor<4x5xf32>) -> ()
  %clone = "poptorch.clone"(%input) : (tensor<4x5xf32>) -> tensor<4x5xf32>

  return %clone : tensor<4x5xf32>
}

// CHECK-LABEL: @clone_source_prewritten(
// CHECK-SAME:                           %[[INPUT:.*]]: tensor<4x5xf32>,
// CHECK-NOT: poptorch.clone
// CHECK: poptorch.overwrite
// CHECK-NOT: poptorch.clone
// CHECK: return %[[INPUT]] : tensor<4x5xf32>


// -----

// Test case: `clone`: write to a view of the source *before* the `clone` is
// taken, so that it can still be removed.

func.func @clone_source_view_prewritten(%input: tensor<4x5xf32>, %other: tensor<5x4xf32>) -> tensor<4x5xf32> {
  %view = "poptorch.view"(%input) {shape = [5, 4]} : (tensor<4x5xf32>) -> tensor<5x4xf32>
  "poptorch.overwrite"(%view, %other) : (tensor<5x4xf32>, tensor<5x4xf32>) -> ()
  %clone = "poptorch.clone"(%input) : (tensor<4x5xf32>) -> tensor<4x5xf32>

  return %clone : tensor<4x5xf32>
}

// CHECK-LABEL: @clone_source_view_prewritten(
// CHECK-SAME:                           %[[INPUT:.*]]: tensor<4x5xf32>,
// CHECK-NOT: poptorch.clone
// CHECK: poptorch.overwrite
// CHECK-NOT: poptorch.clone
// CHECK: return %[[INPUT]] : tensor<4x5xf32>


// -----

// Test case: `clone`: take a view *before* the `clone` is taken, then write to
// it afterwards -- the `clone` should not be removed.

func.func @clone_source_view_postwritten(%input: tensor<4x5xf32>, %other: tensor<5x4xf32>) -> tensor<4x5xf32> {
  %view = "poptorch.view"(%input) {shape = [5, 4]} : (tensor<4x5xf32>) -> tensor<5x4xf32>
  %clone = "poptorch.clone"(%input) : (tensor<4x5xf32>) -> tensor<4x5xf32>
  "poptorch.overwrite"(%view, %other) : (tensor<5x4xf32>, tensor<5x4xf32>) -> ()

  return %clone : tensor<4x5xf32>
}

// CHECK-LABEL: @clone_source_view_postwritten(
// CHECK-SAME:                                 %[[INPUT:.*]]: tensor<4x5xf32>,
// CHECK: %[[CLONE:.*]] = "poptorch.clone"(%[[INPUT]])
// CHECK: return %[[CLONE]] : tensor<4x5xf32>


// -----

// Test case: `overwrite`: if the *source* is a view, remove the `overwrite`
// and replace subsequent uses of the destination's ID.

func.func @overwrite_with_view(%input: tensor<4x5xf32>, %other: tensor<5x4xf32>) -> tensor<5x4xf32> {
  %view = "poptorch.view"(%input) {shape = [5, 4]} : (tensor<4x5xf32>) -> tensor<5x4xf32>

  // To check that pre-`overwrite` uses of `%other` aren't replaced.
  "some.op"(%other) : (tensor<5x4xf32>) -> ()

  "poptorch.overwrite"(%other, %view) : (tensor<5x4xf32>, tensor<5x4xf32>) -> ()

  return %other : tensor<5x4xf32>
}

// CHECK-LABEL: @overwrite_with_view(
// CHECK-SAME:                       %[[INPUT:.*]]: tensor<4x5xf32>
// CHECK-SAME:                       %[[OTHER:.*]]: tensor<5x4xf32>
// CHECK-NOT: poptorch.overwrite
// CHECK: %[[VIEW:.*]] = "poptorch.view"(%[[INPUT]])
// CHECK-NOT: poptorch.overwrite
// CHECK: "some.op"(%[[OTHER]])
// CHECK-NOT: poptorch.overwrite
// CHECK: return %[[VIEW]] : tensor<5x4xf32>


// -----

// Test case: `overwrite`: source is not a view, so don't remove.

func.func @overwrite_without_view(%input: tensor<4x5xf32>, %other: tensor<4x5xf32>) -> tensor<4x5xf32> {
  "poptorch.overwrite"(%input, %other) : (tensor<4x5xf32>, tensor<4x5xf32>) -> ()
  return %input : tensor<4x5xf32>
}

// CHECK-LABEL: @overwrite_without_view(
// CHECK-SAME:                          %[[INPUT:.*]]: tensor<4x5xf32>,
// CHECK-SAME:                          %[[OTHER:.*]]: tensor<4x5xf32>
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[OTHER]]
// CHECK: return %[[INPUT]] : tensor<4x5xf32>


// -----

// Test case: `overwrite`: destination is a view, but not source, so don't
// remove.

func.func @overwrite_destination_view(%input: tensor<4x5xf32>, %other: tensor<5x4xf32>) -> tensor<5x4xf32> {
  %view = "poptorch.view"(%input) {shape = [5, 4]} : (tensor<4x5xf32>) -> tensor<5x4xf32>
  "poptorch.overwrite"(%view, %other) : (tensor<5x4xf32>, tensor<5x4xf32>) -> ()

  return %view : tensor<5x4xf32>
}

// CHECK-LABEL: @overwrite_destination_view(
// CHECK-SAME:                              %[[INPUT:.*]]: tensor<4x5xf32>
// CHECK-SAME:                              %[[OTHER:.*]]: tensor<5x4xf32>
// CHECK: %[[VIEW:.*]] = "poptorch.view"(%[[INPUT]]
// CHECK: poptorch.overwrite replace %[[VIEW]] with %[[OTHER]]
// CHECK: return %[[VIEW]] : tensor<5x4xf32>


// -----

// Test case: `overwrite`: source & destination are views, so remove.

func.func @overwrite_src_and_dest_view(%input: tensor<4x5xf32>, %other: tensor<4x5xf32>) -> tensor<5x4xf32> {
  %input_view = "poptorch.view"(%input) {shape = [5, 4]} : (tensor<4x5xf32>) -> tensor<5x4xf32>
  %other_view = "poptorch.view"(%other) {shape = [5, 4]} : (tensor<4x5xf32>) -> tensor<5x4xf32>
  "poptorch.overwrite"(%input_view, %other_view) : (tensor<5x4xf32>, tensor<5x4xf32>) -> ()

  return %input_view : tensor<5x4xf32>
}

// CHECK-LABEL: @overwrite_src_and_dest_view(
// CHECK-SAME:                               %[[INPUT:.*]]: tensor<4x5xf32>,
// CHECK-SAME:                               %[[OTHER:.*]]: tensor<4x5xf32>
// CHECK-NOT: poptorch.overwrite
// CHECK: %[[OTHER_VIEW:.*]] = "poptorch.view"(%[[OTHER]]
// CHECK: return %[[OTHER_VIEW]] : tensor<5x4xf32>
