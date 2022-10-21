// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// RUN: poptorch-opt --outplace-view-ops %s --split-input-file -allow-unregistered-dialect | FileCheck %s


// Test case: simple view to be outplaced.

// When it encounters the result of a view op being passed to another op (that
// isn't an overwrite), the pass should:
//
// * Insert an outplace variant of the view op.
// * Replace the found usage of the view with the result of the outplace view.
// * Leave the original view op to get DCE'd.
//
// In this case, the 'other op' is the return.
func.func @simple_view(%input: tensor<3xsi32>) -> tensor<3x1xsi32> {
  %view = "poptorch.view"(%input) {shape = [3, 1]} : (tensor<3xsi32>) -> tensor<3x1xsi32>
  return %view : tensor<3x1xsi32>
}

// CHECK-LABEL: @simple_view(
// CHECK-SAME:               %[[INPUT:.*]]: tensor<3xsi32>)
//
// CHECK-NOT: "poptorch.view"(
// CHECK: %[[VIEW:.*]] = "poptorch.viewOutplace"(%[[INPUT]]) {shape = [3, 1]}
// CHECK-NOT: "poptorch.view"(
// CHECK: return %[[VIEW]] : tensor<3x1xsi32>


// -----

// Test case: simple view to be outplaced, in two places.

// If there are *multiple* non-overwrite uses of the same view, we should
// recognise and handle that.
func.func @two_uses_of_view(%input: tensor<3xsi32>) -> (tensor<3x1xsi32>, tensor<3x1xsi32>) {
  %view = "poptorch.view"(%input) {shape = [3, 1]} : (tensor<3xsi32>) -> tensor<3x1xsi32>
  %used_view = "some.op"(%view) : (tensor<3x1xsi32>) -> tensor<3x1xsi32>
  return %view, %used_view : tensor<3x1xsi32>, tensor<3x1xsi32>
}

// CHECK-LABEL: @two_uses_of_view(
// CHECK-SAME:                    %[[INPUT:.*]]: tensor<3xsi32>)
//
// CHECK: %[[VIEW_TO_USE:.*]] = "poptorch.viewOutplace"(%[[INPUT]]) {shape = [3, 1]}
// CHECK: %[[USED_VIEW:.*]] = "some.op"(%[[VIEW_TO_USE]])
// CHECK: %[[VIEW_TO_RETURN:.*]] = "poptorch.viewOutplace"(%[[INPUT]]) {shape = [3, 1]}
// CHECK: return %[[VIEW_TO_RETURN]], %[[USED_VIEW]]


// -----

// Test case: take a view, then put the source tensor through a subsequent inplace op.

// When the pass replaces a view op with its outplace variant, the outplace
// view op should be just before its usage in the graph. Here, its usage is in
// the `return`, so it should occur *after* the `poptorch.overwrite`.
func.func @modify_source(%input: tensor<3xsi32>, %other: tensor<3xsi32>) -> (tensor<3x1xsi32>) {
  %view = "poptorch.view"(%input) {shape = [3, 1]} : (tensor<3xsi32>) -> tensor<3x1xsi32>
  "poptorch.overwrite"(%input, %other) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  return %view : tensor<3x1xsi32>
}

// CHECK-LABEL: @modify_source(
// CHECK-SAME:                 %[[INPUT:.*]]: tensor<3xsi32>, %[[OTHER:.*]]: tensor<3xsi32>)
//
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[OTHER]]
// CHECK: %[[VIEW:.*]] = "poptorch.viewOutplace"(%[[INPUT]])
// CHECK: return %[[VIEW]] : tensor<3x1xsi32>


// -----

// Test case: take a view, then put the view through a subsequent inplace op.

// When the pass encounters a view being passed to an overwrite, it should:
//
// * Create the equivalent inverse view op, just before the overwrite. The
//   inverse should be passed the view tensor, followed by the arguments to the
//   original view op.
// * Create new overwrite ops, that overwrite the original (pre-view) tensors
//   with the results of the inverse view of the modified tensor.
// * Remove the original overwrite.
func.func @modify_view(%input: tensor<3xsi32>, %other: tensor<3x1xsi32>) -> (tensor<3xsi32>) {
  %view = "poptorch.view"(%input) {shape = [3, 1]} : (tensor<3xsi32>) -> tensor<3x1xsi32>
  "poptorch.overwrite"(%view, %other) : (tensor<3x1xsi32>, tensor<3x1xsi32>) -> ()
  return %input : tensor<3xsi32>
}

// CHECK-LABEL: @modify_view(
// CHECK-SAME:               %[[INPUT:.*]]: tensor<3xsi32>, %[[OTHER:.*]]: tensor<3x1xsi32>)
//
// CHECK: %[[MODIFIED_INPUT:.*]] = "poptorch.viewInverse"(%[[OTHER]], %[[INPUT]])
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[MODIFIED_INPUT]]
// CHECK: return %[[INPUT]] : tensor<3xsi32>


// -----

// Test case: take a view, then put it through an outplace op, then an inplace op.

func.func @outplace_then_inplace(%input: tensor<3xsi32>,
                                 %view_mod: tensor<3x1xsi32>) -> (tensor<3x1xsi32>) {
  %view = "poptorch.view"(%input) {shape = [3, 1]} : (tensor<3xsi32>) -> tensor<3x1xsi32>
  "some.op"(%view) : (tensor<3x1xsi32>) -> ()
  "poptorch.overwrite"(%view, %view_mod) : (tensor<3x1xsi32>, tensor<3x1xsi32>) -> ()
  return %view : tensor<3x1xsi32>
}

// CHECK-LABEL: @outplace_then_inplace(
// CHECK-SAME:                          %[[INPUT:.*]]: tensor<3xsi32>, %[[VIEW_MOD:.*]]: tensor<3x1xsi32>)
//
// CHECK: %[[VIEW_SOME_OP:.*]] = "poptorch.viewOutplace"(%[[INPUT]])
// CHECK: "some.op"(%[[VIEW_SOME_OP]])
// CHECK: %[[MODIFIED_INPUT:.*]] = "poptorch.viewInverse"(%[[VIEW_MOD]], %[[INPUT]])
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[MODIFIED_INPUT]]
// CHECK: %[[VIEW_RETURN:.*]] = "poptorch.viewOutplace"(%[[INPUT]])
// CHECK: return %[[VIEW_RETURN]] : tensor<3x1xsi32>


// -----

// Test case: take a view, then put it through an inplace op, then an outplace op.

func.func @inplace_then_outplace(%input: tensor<3xsi32>,
                                 %view_mod: tensor<3x1xsi32>) -> (tensor<3x1xsi32>) {
  %view = "poptorch.view"(%input) {shape = [3, 1]} : (tensor<3xsi32>) -> tensor<3x1xsi32>
  "poptorch.overwrite"(%view, %view_mod) : (tensor<3x1xsi32>, tensor<3x1xsi32>) -> ()
  "some.op"(%view) : (tensor<3x1xsi32>) -> ()
  return %view : tensor<3x1xsi32>
}

// CHECK-LABEL: @inplace_then_outplace(
// CHECK-SAME:                          %[[INPUT:.*]]: tensor<3xsi32>, %[[VIEW_MOD:.*]]: tensor<3x1xsi32>)
//
// CHECK: %[[MODIFIED_INPUT:.*]] = "poptorch.viewInverse"(%[[VIEW_MOD]], %[[INPUT]])
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[MODIFIED_INPUT]]
// CHECK: %[[VIEW_SOME_OP:.*]] = "poptorch.viewOutplace"(%[[INPUT]])
// CHECK: "some.op"(%[[VIEW_SOME_OP]])
// CHECK: %[[VIEW_RETURN:.*]] = "poptorch.viewOutplace"(%[[INPUT]])
// CHECK: return %[[VIEW_RETURN]] : tensor<3x1xsi32>


// -----

// Test case: take a view, then modify the view, then modify the source.

// This is a combination of the parts of the view op handling that replace
// views with their outplace variants, and the parts that respond to overwrites
// by adding inverse variants.
func.func @modify_view_and_source(%input: tensor<3xsi32>,
                                  %input_mod: tensor<3xsi32>,
                                  %view_mod: tensor<3x1xsi32>) -> (tensor<3x1xsi32>) {
  %view = "poptorch.view"(%input) {shape = [3, 1]} : (tensor<3xsi32>) -> tensor<3x1xsi32>
  "poptorch.overwrite"(%view, %view_mod) : (tensor<3x1xsi32>, tensor<3x1xsi32>) -> ()
  "poptorch.overwrite"(%input, %input_mod) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  return %view : tensor<3x1xsi32>
}

// CHECK-LABEL: @modify_view_and_source(
// CHECK-SAME:                          %[[INPUT:.*]]: tensor<3xsi32>, %[[INPUT_MOD:.*]]: tensor<3xsi32>, %[[VIEW_MOD:.*]]: tensor<3x1xsi32>)
//
// CHECK: %[[MODIFIED_INPUT:.*]] = "poptorch.viewInverse"(%[[VIEW_MOD]], %[[INPUT]])
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[MODIFIED_INPUT]]
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[INPUT_MOD]]
// CHECK: %[[VIEW:.*]] = "poptorch.viewOutplace"(%[[INPUT]])
// CHECK: return %[[VIEW]] : tensor<3x1xsi32>


// -----

// Test case: take a view, then modify the source, then modify the view.

func.func @modify_source_and_view(%input: tensor<3xsi32>,
                                  %input_mod: tensor<3xsi32>,
                                  %view_mod: tensor<3x1xsi32>) -> (tensor<3x1xsi32>) {
  %view = "poptorch.view"(%input) {shape = [3, 1]} : (tensor<3xsi32>) -> tensor<3x1xsi32>
  "poptorch.overwrite"(%input, %input_mod) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  "poptorch.overwrite"(%view, %view_mod) : (tensor<3x1xsi32>, tensor<3x1xsi32>) -> ()
  return %view : tensor<3x1xsi32>
}

// CHECK-LABEL: @modify_source_and_view(
// CHECK-SAME:                          %[[INPUT:.*]]: tensor<3xsi32>, %[[INPUT_MOD:.*]]: tensor<3xsi32>, %[[VIEW_MOD:.*]]: tensor<3x1xsi32>)
//
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[INPUT_MOD]]
// CHECK: %[[MODIFIED_INPUT:.*]] = "poptorch.viewInverse"(%[[VIEW_MOD]], %[[INPUT]])
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[MODIFIED_INPUT]]
// CHECK: %[[VIEW:.*]] = "poptorch.viewOutplace"(%[[INPUT]])
// CHECK: return %[[VIEW]] : tensor<3x1xsi32>


// -----

// Test case: take a view of a view.

// This is should be handled similarly to the first test case; the outplacing
// of the outer view should follow from the outplacing of the inner view, due
// to the `return` statement.
func.func @nested_views(%input: tensor<3xsi32>) -> (tensor<1x3xsi32>) {
  %outer_view = "poptorch.view"(%input) {shape = [3, 1]} : (tensor<3xsi32>) -> tensor<3x1xsi32>
  %inner_view = "poptorch.view"(%outer_view) {shape = [1, 3]} : (tensor<3x1xsi32>) -> tensor<1x3xsi32>
  return %inner_view : tensor<1x3xsi32>
}

// CHECK-LABEL: @nested_views(
// CHECK-SAME:                %[[INPUT:.*]]: tensor<3xsi32>)
//
// CHECK: %[[OUTER_VIEW:.*]] = "poptorch.viewOutplace"(%[[INPUT]]) {shape = [3, 1]}
// CHECK: %[[INNER_VIEW:.*]] = "poptorch.viewOutplace"(%[[OUTER_VIEW]]) {shape = [1, 3]}
// CHECK: return %[[INNER_VIEW]] : tensor<1x3xsi32>


// -----

// Test case: take a view, modify it, then take a view of the (modified) view

func.func @view_mod_view(%input: tensor<3xsi32>,
                         %outer_view_mod: tensor<3x1xsi32>) -> (tensor<1x3xsi32>) {
  %outer_view = "poptorch.view"(%input) {shape = [3, 1]} : (tensor<3xsi32>) -> tensor<3x1xsi32>
  "poptorch.overwrite"(%outer_view, %outer_view_mod) : (tensor<3x1xsi32>, tensor<3x1xsi32>) -> ()
  %inner_view = "poptorch.view"(%outer_view) {shape = [1, 3]} : (tensor<3x1xsi32>) -> tensor<1x3xsi32>
  return %inner_view : tensor<1x3xsi32>
}

// CHECK-LABEL: @view_mod_view(
// CHECK-SAME:                 %[[INPUT:.*]]: tensor<3xsi32>, %[[OUTER_VIEW_MOD:.*]]: tensor<3x1xsi32>)
//
// From "poptorch.overwrite"(%outer_view, %outer_view_mod)
// CHECK: %[[MODIFIED_INPUT:.*]] = "poptorch.viewInverse"(%[[OUTER_VIEW_MOD]], %[[INPUT]])
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[MODIFIED_INPUT]]
//
// From return %inner_view
// CHECK: %[[OUTER_VIEW:.*]] = "poptorch.viewOutplace"(%[[INPUT]]) {shape = [3, 1]}
// CHECK: %[[INNER_VIEW:.*]] = "poptorch.viewOutplace"(%[[OUTER_VIEW]]) {shape = [1, 3]}
//
// CHECK: return %[[INNER_VIEW]] : tensor<1x3xsi32>


// -----

// Test case: take a view, modify it twice, then take a view of the (modified) view

func.func @view_two_mods_view(%input: tensor<3xsi32>,
                              %view_mod_1: tensor<3x1xsi32>,
                              %view_mod_2: tensor<3x1xsi32>) -> (tensor<1x3xsi32>) {
  %outer_view = "poptorch.view"(%input) {shape = [3, 1]} : (tensor<3xsi32>) -> tensor<3x1xsi32>
  "poptorch.overwrite"(%outer_view, %view_mod_1) : (tensor<3x1xsi32>, tensor<3x1xsi32>) -> ()
  "poptorch.overwrite"(%outer_view, %view_mod_2) : (tensor<3x1xsi32>, tensor<3x1xsi32>) -> ()
  %inner_view = "poptorch.view"(%outer_view) {shape = [1, 3]} : (tensor<3x1xsi32>) -> tensor<1x3xsi32>
  return %inner_view : tensor<1x3xsi32>
}

// CHECK-LABEL: @view_two_mods_view(
// CHECK-SAME:                      %[[INPUT:.*]]: tensor<3xsi32>, %[[VIEW_MOD_1:.*]]: tensor<3x1xsi32>, %[[VIEW_MOD_2:.*]]: tensor<3x1xsi32>)
//
// From "poptorch.overwrite"(%outer_view, %view_mod_1)
// CHECK: %[[MODIFIED_INPUT_1:.*]] = "poptorch.viewInverse"(%[[VIEW_MOD_1]], %[[INPUT]])
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[MODIFIED_INPUT_1]]
//
// From "poptorch.overwrite"(%outer_view, %view_mod_2)
// CHECK: %[[MODIFIED_INPUT_2:.*]] = "poptorch.viewInverse"(%[[VIEW_MOD_2]], %[[INPUT]])
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[MODIFIED_INPUT_2]]
//
// From return %inner_view
// CHECK: %[[OUTER_VIEW:.*]] = "poptorch.viewOutplace"(%[[INPUT]]) {shape = [3, 1]}
// CHECK: %[[INNER_VIEW:.*]] = "poptorch.viewOutplace"(%[[OUTER_VIEW]]) {shape = [1, 3]}
//
// CHECK: return %[[INNER_VIEW]] : tensor<1x3xsi32>


// -----

// Test case: take a view of a view, then modify the outer view.

func.func @modify_outer_view(%input: tensor<3xsi32>,
                             %outer_view_mod: tensor<3x1xsi32>) -> (tensor<1x3xsi32>) {
  %outer_view = "poptorch.view"(%input) {shape = [3, 1]} : (tensor<3xsi32>) -> tensor<3x1xsi32>
  %inner_view = "poptorch.view"(%outer_view) {shape = [1, 3]} : (tensor<3x1xsi32>) -> tensor<1x3xsi32>
  "poptorch.overwrite"(%outer_view, %outer_view_mod) : (tensor<3x1xsi32>, tensor<3x1xsi32>) -> ()
  return %inner_view : tensor<1x3xsi32>
}

// CHECK-LABEL: @modify_outer_view(
// CHECK-SAME:                     %[[INPUT:.*]]: tensor<3xsi32>, %[[OUTER_VIEW_MOD:.*]]: tensor<3x1xsi32>)
//
// CHECK: %[[MODIFIED_INPUT:.*]] = "poptorch.viewInverse"(%[[OUTER_VIEW_MOD]], %[[INPUT]])
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[MODIFIED_INPUT]]
//
// CHECK: %[[OUTER_VIEW:.*]] = "poptorch.viewOutplace"(%[[INPUT]]) {shape = [3, 1]}
// CHECK: %[[INNER_VIEW:.*]] = "poptorch.viewOutplace"(%[[OUTER_VIEW]]) {shape = [1, 3]}
//
// CHECK: return %[[INNER_VIEW]] : tensor<1x3xsi32>


// -----

// Test case: take a view of a view, then modify the inner view.

// We should see:
// * overwrite of %inner_view -> replace overwrite with viewInverse + overwrite of %outer_view
// * overwrite of %outer_view -> replace overwrite with viewInverse + overwrite of %input
// * use of view of %outer_view in `return` -> replace view of %outer_view with viewOutplace, at end
// * two uses of a view of %input:
//     * in viewInverse, near start
//     * in viewOutplace, near end
// * ...insert two viewOutplaces for these uses, then DCE the original view(%input) op
func.func @modify_inner_view(%input: tensor<3xsi32>,
                             %inner_view_mod: tensor<1x3xsi32>) -> (tensor<1x3xsi32>) {
  %outer_view = "poptorch.view"(%input) {shape = [3, 1]} : (tensor<3xsi32>) -> tensor<3x1xsi32>
  %inner_view = "poptorch.view"(%outer_view) {shape = [1, 3]} : (tensor<3x1xsi32>) -> tensor<1x3xsi32>
  "poptorch.overwrite"(%inner_view, %inner_view_mod) : (tensor<1x3xsi32>, tensor<1x3xsi32>) -> ()
  return %inner_view : tensor<1x3xsi32>
}

// CHECK-LABEL: @modify_inner_view(
// CHECK-SAME:                     %[[INPUT:.*]]: tensor<3xsi32>, %[[INNER_VIEW_MOD:.*]]: tensor<1x3xsi32>)
//
// From "poptorch.overwrite"(%inner_view, %inner_view_mod)
// CHECK: %[[OUTER_VIEW_PRE:.*]] = "poptorch.viewOutplace"(%[[INPUT]]) {shape = [3, 1]}
// CHECK: %[[MODIFIED_OUTER:.*]] = "poptorch.viewInverse"(%[[INNER_VIEW_MOD]], %[[OUTER_VIEW_PRE]])
// CHECK: %[[MODIFIED_INPUT:.*]] = "poptorch.viewInverse"(%[[MODIFIED_OUTER]], %[[INPUT]])
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[MODIFIED_INPUT]]
//
// From return %inner_view
// CHECK: %[[OUTER_VIEW:.*]] = "poptorch.viewOutplace"(%[[INPUT]]) {shape = [3, 1]}
// CHECK: %[[INNER_VIEW:.*]] = "poptorch.viewOutplace"(%[[OUTER_VIEW]]) {shape = [1, 3]}
//
// CHECK: return %[[INNER_VIEW]] : tensor<1x3xsi32>


// -----

// Test case: take a view of a view, and modify both the inner and outer views, as well as the source.

func.func @modify_nested_views(%input: tensor<3xsi32>,
                               %input_mod: tensor<3xsi32>,
                               %outer_view_mod: tensor<3x1xsi32>,
                               %inner_view_mod: tensor<1x3xsi32>) -> (tensor<1x3xsi32>) {
  %outer_view = "poptorch.view"(%input) {shape = [3, 1]} : (tensor<3xsi32>) -> tensor<3x1xsi32>
  %inner_view = "poptorch.view"(%outer_view) {shape = [1, 3]} : (tensor<3x1xsi32>) -> tensor<1x3xsi32>
  "poptorch.overwrite"(%input, %input_mod) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  "poptorch.overwrite"(%outer_view, %outer_view_mod) : (tensor<3x1xsi32>, tensor<3x1xsi32>) -> ()
  "poptorch.overwrite"(%inner_view, %inner_view_mod) : (tensor<1x3xsi32>, tensor<1x3xsi32>) -> ()
  return %inner_view : tensor<1x3xsi32>
}

// CHECK-LABEL: @modify_nested_views(
// CHECK-SAME:                       %[[INPUT:.*]]: tensor<3xsi32>, %[[INPUT_MOD:.*]]: tensor<3xsi32>, %[[OUTER_VIEW_MOD:.*]]: tensor<3x1xsi32>, %[[INNER_VIEW_MOD:.*]]: tensor<1x3xsi32>)
//
// From "poptorch.overwrite"(%input, %input_mod)
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[INPUT_MOD]]
//
// From "poptorch.overwrite"(%outer_view, %outer_view_mod)
// CHECK: %[[MODIFIED_INPUT:.*]] = "poptorch.viewInverse"(%[[OUTER_VIEW_MOD]], %[[INPUT]])
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[MODIFIED_INPUT]]
//
// From "poptorch.overwrite(%inner_view, %inner_view_mod)
// CHECK: %[[OUTER_VIEW_PRE:.*]] = "poptorch.viewOutplace"(%[[INPUT]]) {shape = [3, 1]}
// CHECK: %[[MODIFIED_OUTER:.*]] = "poptorch.viewInverse"(%[[INNER_VIEW_MOD]], %[[OUTER_VIEW_PRE]])
// CHECK: %[[MODIFIED_INPUT_2:.*]] = "poptorch.viewInverse"(%[[MODIFIED_OUTER]], %[[INPUT]])
// CHECK: poptorch.overwrite replace %[[INPUT]] with %[[MODIFIED_INPUT_2]]
//
// From return %inner_view
// CHECK: %[[OUTER_VIEW:.*]] = "poptorch.viewOutplace"(%[[INPUT]]) {shape = [3, 1]}
// CHECK: %[[INNER_VIEW:.*]] = "poptorch.viewOutplace"(%[[OUTER_VIEW]]) {shape = [1, 3]}
//
// CHECK: return %[[INNER_VIEW]] : tensor<1x3xsi32>
