#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import pytest
import poptorch


@pytest.mark.parametrize("use_half", [True, False])
def test_simple_tuple(use_half):
    class SimpleAdder(nn.Module):
        def forward(self, t):
            assert isinstance(t, tuple)
            (x, y) = t
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            return x + y

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])

    if use_half:
        model.half()
        t1 = t1.half()
        t2 = t2.half()
    assert inference_model((t1, t2)).float() == 3.0


@pytest.mark.parametrize("use_half", [True, False])
def test_nested_tuples(use_half):
    class SimpleAdder(nn.Module):
        def forward(self, tpl1, t2, tpl34567):
            (t1, ) = tpl1
            (t3, (t4, t5), _) = tpl34567
            (t6, _) = tpl34567[2]
            t7 = tpl34567[2][1]

            assert isinstance(t1, torch.Tensor)
            assert isinstance(t2, torch.Tensor)
            assert isinstance(t3, torch.Tensor)
            assert isinstance(t4, torch.Tensor)
            assert isinstance(t5, torch.Tensor)
            assert isinstance(t6, torch.Tensor)
            assert isinstance(t7, torch.Tensor)

            return t1 + t2 + t3 + t4 + t5 + t6 + t7

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])
    t3 = torch.tensor([3.])
    t4 = torch.tensor([4.], dtype=torch.float64)
    t5 = torch.tensor([5.])
    t6 = torch.tensor([6.])
    t7 = torch.tensor([7.], dtype=torch.float64)

    if use_half:
        model.half()
        t1 = t1.half()
        t2 = t2.half()
        t3 = t3.half()
        t4 = t4.half()
        t5 = t5.half()
        t6 = t6.half()
        t7 = t7.half()
    assert inference_model((t1, ), t2,
                           (t3, (t4, t5), (t6, t7))).float() == 28.0


@pytest.mark.parametrize("use_half", [True, False])
def test_optional_inputs(use_half):
    dtype = torch.float16 if use_half else torch.float32

    class SimpleAdder(nn.Module):
        def forward(self,
                    t1,
                    t2,
                    t3=torch.ones(1, dtype=dtype),
                    t4=torch.zeros(1, dtype=dtype)):
            return t1 * t3 + t2 * t4

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])
    t4 = torch.tensor([4.])

    if use_half:
        model.half()
        t1 = t1.half()
        t2 = t2.half()
        t4 = t4.half()

    assert inference_model(t1, t2).float() == 1.0
    assert inference_model(t1, t2, t4=t4).float() == 9.0
    assert inference_model(t4=t4, t1=t1, t2=t2).float() == 9.0


@pytest.mark.parametrize("use_half", [True, False])
def test_list_inputs(use_half):
    class SimpleAdder(nn.Module):
        def forward(self, t1, t2, x):
            l = [t1, t2]
            x = l[0] + x
            l[1] = x
            return l

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])
    t3 = torch.tensor([4.])

    if use_half:
        model.half()
        t1 = t1.half()
        t2 = t2.half()
        t3 = t3.half()

    expected = [torch.tensor([1.0]), torch.tensor([5.0])]

    assert [t.float() for t in inference_model(t1, t2, t3)] == expected


def test_unused_tuple():
    class SimpleAdder(nn.Module):
        def forward(self, x, y, z):  # pylint: disable=unused-argument
            return x + y

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)
    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])
    z = (torch.tensor([1.]), torch.tensor([1.]))
    inference_model(t1, t2, z)


def test_input_plain_list():
    model = torch.nn.ReLU()
    inference_model = poptorch.inferenceModel(model)

    with pytest.raises(TypeError) as excinfo:
        inference_model([torch.tensor([1])])

    assert (str(
        excinfo.value) == "Lists are not supported as input arguments, "
            "including when nested in tuples.\n"
            "Received list input = [tensor([1])]")


def test_input_nested_list():
    model = torch.nn.ReLU()
    inference_model = poptorch.inferenceModel(model)

    with pytest.raises(TypeError) as excinfo:
        inference_model((torch.tensor([1]), torch.tensor([2]),
                         (torch.tensor([3]), [torch.tensor([4])],
                          torch.tensor([5])), torch.tensor([6])))

    assert (str(
        excinfo.value) == "Lists are not supported as input arguments, "
            "including when nested in tuples.\n"
            "Received list input[2][1] = [tensor([4])]")


def test_input_three_agg_nested_list():
    class ThreeIdentity(nn.Module):
        def forward(self, x, y, z):
            return x, y, z

    model = ThreeIdentity()
    inference_model = poptorch.inferenceModel(model)

    with pytest.raises(TypeError) as excinfo:
        inference_model(torch.tensor([0]),
                        (torch.tensor([1]), torch.tensor([2]),
                         (torch.tensor([3]), [torch.tensor(
                             [4])], torch.tensor([5])), torch.tensor([6])),
                        torch.tensor([7]))

    assert (str(
        excinfo.value) == "Lists are not supported as input arguments, "
            "including when nested in tuples.\n"
            "Received list y[2][1] = [tensor([4])]")


def test_input_plain_dict():
    model = torch.nn.ReLU()
    inference_model = poptorch.inferenceModel(model)

    with pytest.raises(TypeError) as excinfo:
        inference_model({'a': torch.tensor([1])})

    assert (str(
        excinfo.value) == "Dictionaries are not supported as input arguments, "
            "including when nested in tuples.\n"
            "Received dict input = {'a': tensor([1])}")


def test_input_nested_dict():
    model = torch.nn.ReLU()
    inference_model = poptorch.inferenceModel(model)

    with pytest.raises(TypeError) as excinfo:
        inference_model(
            (torch.tensor([1]), torch.tensor([2]), (torch.tensor([3]), {
                'b': torch.tensor([4])
            }, torch.tensor([5])), torch.tensor([6])))

    assert (str(excinfo.value) == "Dictionaries are not supported as input "
            "arguments, including when nested in tuples."
            "\nReceived dict input[2][1] = "
            "{'b': tensor([4])}")


def test_input_three_agg_nested_dict():
    class ThreeIdentity(nn.Module):
        def forward(self, x, y, z):
            return x, y, z

    model = ThreeIdentity()

    inference_model = poptorch.inferenceModel(model)

    with pytest.raises(TypeError) as excinfo:
        inference_model(torch.tensor([0]),
                        (torch.tensor([1]), torch.tensor([2]),
                         (torch.tensor([3]), {
                             'c': torch.tensor([4])
                         }, torch.tensor([5])), torch.tensor([6])),
                        torch.tensor([7]))

    assert (str(excinfo.value) == "Dictionaries are not supported as input "
            "arguments, including when nested in tuples."
            "\nReceived dict y[2][1] = "
            "{'c': tensor([4])}")
