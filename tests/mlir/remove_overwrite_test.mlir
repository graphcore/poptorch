// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// RUN: poptorch-opt --remove-overwrite %s --split-input-file -allow-unregistered-dialect | FileCheck %s


// Test case: Simple case of deleting an overwrite.

func.func @simple_overwrite(%input: tensor<3xsi32>, %other: tensor<3xsi32>) -> tensor<3xsi32> {
  "poptorch.overwrite"(%input, %other) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  return %input : tensor<3xsi32>
}

// CHECK-LABEL: @simple_overwrite(
// CHECK-SAME:                    %[[INPUT:.*]]: tensor<3xsi32>, %[[OTHER:.*]]: tensor<3xsi32>
// CHECK-NOT: poptorch.overwrite
// CHECK: return %[[OTHER]]


// -----

// Test case: Overwrite is between two different dtypes, so should cast.

func.func @overwrite_cast(%input: tensor<3xsi32>, %other: tensor<3xsi16>) -> tensor<3xsi32> {
  "poptorch.overwrite"(%input, %other) : (tensor<3xsi32>, tensor<3xsi16>) -> ()
  return %input : tensor<3xsi32>
}

// CHECK-LABEL: @overwrite_cast(
// CHECK-SAME:                  %[[INPUT:.*]]: tensor<3xsi32>, %[[OTHER:.*]]: tensor<3xsi16>
// CHECK-NOT: poptorch.overwrite
// CHECK: %[[OVERWRITTEN:.*]] = "poptorch.cast"(%[[OTHER]]) {dtype = si32} : (tensor<3xsi16>) -> tensor<3xsi32>
// CHECK-NOT: poptorch.overwrite
// CHECK: return %[[OVERWRITTEN]]


// -----

// Test case: Overwrite happens after the destination has been through an op.

// Should see that the destination's ID doesn't change before the overwrite.
func.func @overwrite_after_op(%input: tensor<3xsi32>, %other: tensor<3xsi32>) -> tensor<3xsi32> {
  "some.op"(%input) : (tensor<3xsi32>) -> ()
  "poptorch.overwrite"(%input, %other) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  return %input : tensor<3xsi32>
}

// CHECK-LABEL: @overwrite_after_op(
// CHECK-SAME:                      %[[INPUT:.*]]: tensor<3xsi32>, %[[OTHER:.*]]: tensor<3xsi32>
// CHECK: "some.op"(%[[INPUT]])
// CHECK: return %[[OTHER]]


// -----

// Test case: Two overwrites onto the same tensor -- check that the final one's
// used.

func.func @parallel_overwrites(%input: tensor<3xsi32>,
                               %other_1: tensor<3xsi32>,
                               %other_2: tensor<3xsi32>) -> tensor<3xsi32> {
  "poptorch.overwrite"(%input, %other_1) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  "poptorch.overwrite"(%input, %other_2) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  return %input : tensor<3xsi32>
}

// CHECK-LABEL: @parallel_overwrites(
// CHECK-SAME:                       %[[INPUT:.*]]: tensor<3xsi32>, %[[OTHER_1:.*]]: tensor<3xsi32>, %[[OTHER_2:.*]]: tensor<3xsi32>
// CHECK-NOT: poptorch.overwrite
// CHECK: return %[[OTHER_2]]


// -----

// Test case: A 'chain' of two overwrites -- check the final ID.

func.func @serial_overwrites(%input: tensor<3xsi32>,
                             %other_1: tensor<3xsi32>,
                             %other_2: tensor<3xsi32>) -> tensor<3xsi32> {
  "poptorch.overwrite"(%other_1, %other_2) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  "poptorch.overwrite"(%input, %other_1) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  return %input : tensor<3xsi32>
}

// CHECK-LABEL: @serial_overwrites(
// CHECK-SAME:                     %[[INPUT:.*]]: tensor<3xsi32>, %[[OTHER_1:.*]]: tensor<3xsi32>, %[[OTHER_2:.*]]: tensor<3xsi32>
// CHECK-NOT: poptorch.overwrite
// CHECK: return %[[OTHER_2]]


// -----

// Test case: Overwrite A over B, then B over A: the second overwrite shouldn't
// do anything.

func.func @inverse_overwrites(%input: tensor<3xsi32>,
                             %other: tensor<3xsi32>) -> tensor<3xsi32> {
  "poptorch.overwrite"(%input, %other) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  "poptorch.overwrite"(%other, %input) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  return %input : tensor<3xsi32>
}

// CHECK-LABEL: @inverse_overwrites(
// CHECK-SAME:                     %[[INPUT:.*]]: tensor<3xsi32>, %[[OTHER:.*]]: tensor<3xsi32>
// CHECK-NOT: poptorch.overwrite
// CHECK: return %[[OTHER]]


// -----

// Test case: A 'chain' of overwrites -- check the intermediate IDs.

func.func @serial_overwrite_intermediates(%input: tensor<3xsi32>,
                                          %other_1: tensor<3xsi32>,
                                          %other_2: tensor<3xsi32>) -> tensor<3xsi32> {
  "some.op"(%input) : (tensor<3xsi32>) -> ()
  "some.op"(%other_1) : (tensor<3xsi32>) -> ()
  "some.op"(%other_2) : (tensor<3xsi32>) -> ()
  "poptorch.overwrite"(%other_1, %other_2) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  "some.op"(%input) : (tensor<3xsi32>) -> ()
  "some.op"(%other_1) : (tensor<3xsi32>) -> ()
  "some.op"(%other_2) : (tensor<3xsi32>) -> ()
  "poptorch.overwrite"(%input, %other_1) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  "some.op"(%input) : (tensor<3xsi32>) -> ()
  "some.op"(%other_1) : (tensor<3xsi32>) -> ()
  "some.op"(%other_2) : (tensor<3xsi32>) -> ()
  "poptorch.overwrite"(%other_2, %input) : (tensor<3xsi32>, tensor<3xsi32>) -> ()
  "some.op"(%input) : (tensor<3xsi32>) -> ()
  "some.op"(%other_1) : (tensor<3xsi32>) -> ()
  "some.op"(%other_2) : (tensor<3xsi32>) -> ()
  return %input : tensor<3xsi32>
}

// CHECK-LABEL: @serial_overwrite_intermediates(
// CHECK-SAME:                                  %[[INPUT:.*]]: tensor<3xsi32>, %[[OTHER_1:.*]]: tensor<3xsi32>, %[[OTHER_2:.*]]: tensor<3xsi32>
//
// Initially:
// CHECK: "some.op"(%[[INPUT]])
// CHECK: "some.op"(%[[OTHER_1]])
// CHECK: "some.op"(%[[OTHER_2]])
//
// After overwrite(%other_1, %other_2)
// CHECK: "some.op"(%[[INPUT]])
// CHECK: "some.op"(%[[OTHER_2]])
// CHECK: "some.op"(%[[OTHER_2]])
//
// After overwrite(%input, %other_1)
// CHECK: "some.op"(%[[OTHER_2]])
// CHECK: "some.op"(%[[OTHER_2]])
// CHECK: "some.op"(%[[OTHER_2]])
//
// After overwrite(%other_2, %input)
// CHECK: "some.op"(%[[OTHER_2]])
// CHECK: "some.op"(%[[OTHER_2]])
// CHECK: "some.op"(%[[OTHER_2]])
//
// CHECK: return %[[OTHER_2]]


// -----

// Test case: Chain of overwrites that should cast.

func.func @serial_overwrite_casts(%input: tensor<3xsi32>,
                                  %other_1: tensor<3xsi16>,
                                  %other_2: tensor<3xf32>) -> tensor<3xsi32> {
  "poptorch.overwrite"(%other_1, %other_2) : (tensor<3xsi16>, tensor<3xf32>) -> ()
  "poptorch.overwrite"(%input, %other_1) : (tensor<3xsi32>, tensor<3xsi16>) -> ()
  return %input : tensor<3xsi32>
}

// CHECK-LABEL: @serial_overwrite_casts(
// CHECK-SAME:                          %[[INPUT:.*]]: tensor<3xsi32>, %[[OTHER_1:.*]]: tensor<3xsi16>, %[[OTHER_2:.*]]: tensor<3xf32>
// CHECK: %[[OVERWRITTEN_OTHER_1:.*]] = "poptorch.cast"(%[[OTHER_2]]) {dtype = si16} : (tensor<3xf32>) -> tensor<3xsi16>
// CHECK: %[[OVERWRITTEN_INPUT:.*]] = "poptorch.cast"(%[[OVERWRITTEN_OTHER_1]]) {dtype = si32} : (tensor<3xsi16>) -> tensor<3xsi32>
// CHECK: return %[[OVERWRITTEN_INPUT]]
