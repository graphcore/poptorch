#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import collections
import ctypes
import pathlib
import random
import sys

import pytest
import torch
import poptorch
import helpers

myso = list(pathlib.Path("tests").rglob("libcustom_*.*"))
assert myso, "Failed to find libcustom_* libraries"
for single_so in myso:
    ctypes.cdll.LoadLibrary(single_so)


def test_float_attribute():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = poptorch.custom_op([x],
                                   "AddScalarFloat",
                                   "test.poptorch",
                                   1,
                                   example_outputs=[x],
                                   attributes={"scalar": 3.5})
            return x

    model = Model()

    x = torch.tensor([5.0])

    inference_model = poptorch.inferenceModel(model)
    out = inference_model(x)
    expected = torch.tensor([8.5])

    helpers.assert_allclose(actual=out[0], expected=expected)


def test_float_attribute_too_low():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = poptorch.custom_op([x],
                                   "AddScalarFloat",
                                   "test.poptorch",
                                   1,
                                   example_outputs=[x],
                                   attributes={"scalar": -sys.float_info.max})
            return x

    model = Model()

    x = torch.tensor([5.0])
    inference_model = poptorch.inferenceModel(model)

    with pytest.raises(
            RuntimeError,
            match=r"-1\.79769e\+308 is too low for a Popart float " +
            r"attribute\."):
        inference_model(x)


def test_float_attribute_too_high():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = poptorch.custom_op([x],
                                   "AddScalarFloat",
                                   "test.poptorch",
                                   1,
                                   example_outputs=[x],
                                   attributes={"scalar": sys.float_info.max})
            return x

    model = Model()

    x = torch.tensor([5.0])
    inference_model = poptorch.inferenceModel(model)

    with pytest.raises(
            RuntimeError,
            match=r"1\.79769e\+308 is too high for a Popart float " +
            r"attribute\."):
        inference_model(x)


def test_int_attribute():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = poptorch.custom_op([x],
                                   "AddScalarInt",
                                   "test.poptorch",
                                   1,
                                   example_outputs=[x],
                                   attributes={"scalar": 3})
            return x

    model = Model()

    x = torch.tensor([5])

    inference_model = poptorch.inferenceModel(model)
    out = inference_model(x)

    helpers.assert_allequal(actual=out[0],
                            expected=torch.tensor([8], dtype=torch.int32))


def test_float_list_attribute():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = poptorch.custom_op([x],
                                   "AddScalarVecFloat",
                                   "test.poptorch",
                                   1,
                                   example_outputs=[x],
                                   attributes={"vec": [1.0, 2.0, 3.0]})
            return x

    model = Model()

    x = torch.tensor([3.0, 4.0, 5.0])

    inference_model = poptorch.inferenceModel(model)
    out = inference_model(x)

    helpers.assert_allclose(actual=out[0],
                            expected=torch.tensor([4.0, 6.0, 8.0]))


def test_float_list_attribute_too_low():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = poptorch.custom_op(
                [x],
                "AddScalarVecFloat",
                "test.poptorch",
                1,
                example_outputs=[x],
                attributes={"vec": [1.0, 2.0, -sys.float_info.max]})
            return x

    model = Model()

    x = torch.tensor([3.0, 4.0, 5.0])

    inference_model = poptorch.inferenceModel(model)
    with pytest.raises(
            RuntimeError,
            match=r"-1\.79769e\+308 is too low for a Popart float " +
            r"attribute\."):
        inference_model(x)


def test_float_list_attribute_too_high():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = poptorch.custom_op(
                [x],
                "AddScalarVecFloat",
                "test.poptorch",
                1,
                example_outputs=[x],
                attributes={"vec": [sys.float_info.max, 2.0, 3.0]})
            return x

    model = Model()

    x = torch.tensor([3.0, 4.0, 5.0])

    inference_model = poptorch.inferenceModel(model)
    with pytest.raises(
            RuntimeError,
            match=r"1\.79769e\+308 is too high for a Popart float " +
            r"attribute\."):
        inference_model(x)


def test_float_tuple_attribute():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = poptorch.custom_op([x],
                                   "AddScalarVecFloat",
                                   "test.poptorch",
                                   1,
                                   example_outputs=[x],
                                   attributes={"vec": (1.0, 2.0, 3.0)})
            return x

    model = Model()

    x = torch.tensor([3.0, 4.0, 5.0])

    inference_model = poptorch.inferenceModel(model)
    out = inference_model(x)

    helpers.assert_allclose(expected=out[0],
                            actual=torch.tensor([4.0, 6.0, 8.0]))


def test_int_list_attribute():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = poptorch.custom_op([x],
                                   "AddScalarVecInt",
                                   "test.poptorch",
                                   1,
                                   example_outputs=[x],
                                   attributes={"vec": [1, 2, 3]})
            return x

    model = Model()

    x = torch.tensor([3, 4, 5])

    inference_model = poptorch.inferenceModel(model)
    out = inference_model(x)

    helpers.assert_allequal(actual=out[0],
                            expected=torch.tensor([4, 6, 8],
                                                  dtype=torch.int32))


def test_float_combined_attributes():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = poptorch.custom_op([x],
                                   "AddVecScalarMulFloat",
                                   "test.poptorch",
                                   1,
                                   example_outputs=[x],
                                   attributes={
                                       "vec": [1.0, 2.0, 3.0],
                                       "scalar": 2.0
                                   })
            return x

    model = Model()

    x = torch.tensor([3.0, 4.0, 5.0])

    inference_model = poptorch.inferenceModel(model)
    out = inference_model(x)

    helpers.assert_allequal(actual=out[0],
                            expected=torch.tensor([8.0, 12.0, 16.0]))


def test_int_two_attributes():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = poptorch.custom_op([x],
                                   "AddScalarInt",
                                   "test.poptorch",
                                   1,
                                   example_outputs=[x],
                                   attributes={"scalar": 3})
            x = poptorch.custom_op(x,
                                   "AddScalarInt",
                                   "test.poptorch",
                                   1,
                                   example_outputs=x,
                                   attributes={"scalar": 2})
            return x

    model = Model()

    x = torch.tensor([5])

    inference_model = poptorch.inferenceModel(model)
    out = inference_model(x)

    helpers.assert_allequal(actual=out[0],
                            expected=torch.tensor([10], dtype=torch.int32))


@pytest.mark.parametrize("attr", ("sum", "mean"))
def test_string_attribute(attr):
    class Model(torch.nn.Module):
        def forward(self, x):
            x = poptorch.custom_op([x],
                                   "ReduceOp",
                                   "test.poptorch",
                                   1,
                                   example_outputs=[x],
                                   attributes={"reduction": attr})
            return x

    model = Model()

    x = torch.tensor([5.0, 6.0, 7.0])

    inference_model = poptorch.inferenceModel(model)
    out = inference_model(x)

    if attr == "mean":
        helpers.assert_allclose(actual=out[0], expected=torch.tensor(6.0))
    else:
        helpers.assert_allclose(actual=out[0], expected=torch.tensor(18.0))


def test_non_ascii_string_attribute():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = poptorch.custom_op([x],
                                   "ReduceOp",
                                   "test.poptorch",
                                   1,
                                   example_outputs=[x],
                                   attributes={"reduction": "a\u1f00b"})
            return x

    model = Model()

    x = torch.tensor([5.0, 6.0, 7.0])

    inference_model = poptorch.inferenceModel(model)

    with pytest.raises(ValueError,
                       match="a\u1f00b contains non-ASCII characters."):
        inference_model(x)


def test_string_list_attribute():
    class Model(torch.nn.Module):
        def forward(self, x, y, z):
            x = poptorch.custom_op(
                [x, y, z],
                "ThreeReduceOp",
                "test.poptorch",
                1,
                example_outputs=[x, y, z],
                attributes={"reductions": ["mean", "sum", "mean"]})
            return x

    model = Model()

    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([2.0, 3.0, 4.0])
    z = torch.tensor([3.0, 4.0, 5.0])

    inference_model = poptorch.inferenceModel(model)
    out = inference_model(x, y, z)

    helpers.assert_allequal(actual=out[0], expected=torch.tensor(2.0))
    helpers.assert_allequal(actual=out[1], expected=torch.tensor(9.0))
    helpers.assert_allequal(actual=out[2], expected=torch.tensor(4.0))


def test_non_asciistring_list_attribute():
    class Model(torch.nn.Module):
        def forward(self, x, y, z):
            x = poptorch.custom_op(
                [x, y, z],
                "ThreeReduceOp",
                "test.poptorch",
                1,
                example_outputs=[x, y, z],
                attributes={"reductions": ["a\u1f00b", "sum", "mean"]})
            return x

    model = Model()

    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([2.0, 3.0, 4.0])
    z = torch.tensor([3.0, 4.0, 5.0])

    inference_model = poptorch.inferenceModel(model)
    with pytest.raises(ValueError,
                       match="a\u1f00b contains non-ASCII characters."):
        inference_model(x, y, z)


ALL_ATTRIBUTES = {
    "float_one": 1.0,
    "float_minus_two": -2.0,
    "int_zero": 0,
    "int_minus_five": -5,
    "floats_one_two_three": [1.0, 2.0, 3.0],
    "floats_minus_one_two_three": [-1.0, -2.0, -3.0],
    "ints_one_two_three": [1, 2, 3],
    "ints_minus_one_two_three": [-1, -2, -3],
    "a_string": "string with quotes and slash \" ' \\ end",
    "strs": ["\x01", "\x02", "\x03"]
}


@pytest.mark.parametrize("seed", range(10))
def test_many_attributes(seed):
    attr_keys = list(ALL_ATTRIBUTES.keys())
    random.seed(seed)
    random.shuffle(attr_keys)
    attrs_shuff = collections.OrderedDict()

    for key in attr_keys:
        attrs_shuff[key] = ALL_ATTRIBUTES[key]

    class Model(torch.nn.Module):
        def forward(self, x):
            x = poptorch.custom_op([x],
                                   "ManyAttributeOp",
                                   "test.poptorch",
                                   1,
                                   example_outputs=[x],
                                   attributes=attrs_shuff)
            return x

    model = Model()

    x = torch.tensor([0.0])

    inference_model = poptorch.inferenceModel(model)
    out = inference_model(x)

    helpers.assert_allequal(actual=out[0],
                            expected=torch.tensor(1.0).reshape((1, )))


@pytest.mark.parametrize("seed", range(3))
def test_many_attributes_one_wrong(seed):
    attr_keys = list(ALL_ATTRIBUTES.keys())
    random.seed(seed)
    random.shuffle(attr_keys)
    attrs_shuff = collections.OrderedDict()

    for key in attr_keys:
        attrs_shuff[key] = ALL_ATTRIBUTES[key]
    attrs_shuff["a_string"] = "Very wrong"

    class Model(torch.nn.Module):
        def forward(self, x):
            x = poptorch.custom_op([x],
                                   "ManyAttributeOp",
                                   "test.poptorch",
                                   1,
                                   example_outputs=[x],
                                   attributes=attrs_shuff)
            return x

    model = Model()

    x = torch.tensor([0.0])

    inference_model = poptorch.inferenceModel(model)
    out = inference_model(x)

    helpers.assert_allequal(actual=out[0],
                            expected=torch.tensor(0.0).reshape((1, )))


#many_attribtes_examples_start
def test_many_attributes_examples():
    class Model(torch.nn.Module):
        def forward(self, x):
            attributes = {
                "float_one": 1.0,
                "float_minus_two": -2.0,
                "int_zero": 0,
                "int_minus_five": -5,
                "floats_one_two_three": [1.0, 2.0, 3.0],
                "floats_minus_one_two_three": [-1.0, -2.0, -3.0],
                "ints_one_two_three": [1, 2, 3],
                "ints_minus_one_two_three": [-1, -2, -3],
                "a_string": "string with quotes and slash \" ' \\ end",
                "strs": ["abc", "def", "ghi"]
            }

            x = poptorch.custom_op([x],
                                   "ManyAttributeOp",
                                   "test.poptorch",
                                   1,
                                   example_outputs=[x],
                                   attributes=attributes)
            #many_attribtes_examples_end
            return x

    model = Model()

    x = torch.tensor([0.0])

    inference_model = poptorch.inferenceModel(model)
    inference_model(x)
