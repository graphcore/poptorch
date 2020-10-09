#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import contextlib
import os
import io
import re
import sys
import pytest

parser = argparse.ArgumentParser(description="Generate CTestTestfile.cmake")
parser.add_argument("test_dir", help="Path to the folder containing the tests")
parser.add_argument("output_file", help="Path to CTestTestfile.cmake")
parser.add_argument("--add_to_sys_path", help="Path to add to sys.path")

args = parser.parse_args()

if args.add_to_sys_path:
    for path in args.add_to_sys_path.split(";"):
        print(f"Adding {path}")
        sys.path.insert(0, path)

# Collect the list of tests:
list_tests = io.StringIO()
with contextlib.redirect_stdout(list_tests):
    retval = pytest.main(["-x", args.test_dir, "--collect-only", "-q"])

assert retval == pytest.ExitCode.OK, f"{str(retval)}: {list_tests.getvalue()}"

# Run all the tests contained in these files in a single process
# because they're small / short to run (Under 1 minute)
#pylint: disable=line-too-long
short_tests = [
    "replicated_graph_test.py", "pipeline_tests_test.py",
    "shape_inference_test.py", "buffers_test.py", "poplar_executor_test.py",
    "ops_test.py", "non_contiguous_tensors_test.py", "options_test.py",
    "outputs_test.py", "custom_loss_test.py", "blas_test.py",
    "custom_ops_test.py", "dataloader_test.py", "inputs_test.py",
    "activations_test.py", "optimizers_test.py", "lstm_test.py",
    "random_sampling_test.py", "batching_test.py", "misc_nn_layers_test.py"
]

long_tests = [
    "torch_nn_test.py::test_pytorch_nn[True-test_nn_Conv2d_circular_stride2_pad2]",
    "bert_small_and_medium_test.py::test_bert_medium_result",
    "torch_nn_test.py::test_pytorch_nn[False-test_nn_Conv2d_circular_stride2_pad2]",
    "resnets_inference_test.py::test_mobilenet_v2",
    "resnets_inference_test.py::test_resnet18",
    "resnets_inference_test.py::test_mnasnet1_0",
    "resnets_inference_test.py::test_resnext50_32x4d",
    "half_test.py::test_resnet", "resnets_inference_test.py::test_googlenet"
]
#pylint: enable=line-too-long


def add_test(output, test, folder, labels):
    output.write(f"add_test({test} \"python3\" \"-m\" \"pytest\" \"-s\" "
                 f"\"{folder}/{test}\")\n")
    if labels:
        output.write(f"set_tests_properties({test} PROPERTIES  LABELS "
                     f"\"{labels}\")\n")


with open(args.output_file, "w") as output:
    # Add the short_tests files
    for test in short_tests:
        add_test(output, test, args.test_dir, "short")

    # Process the list of tests returned by pytest
    for test in list_tests.getvalue().split("\n"):
        # Extract the file name from the test name
        m = re.match("^(.*)::(.*)", test)
        if m:
            # Use os.path.basename() to ensure we only have
            # the filename
            test_file = os.path.basename(m.group(1))
            if test_file in short_tests:
                continue
            labels = ""
            if test in long_tests:
                labels = "long"
            add_test(output, f"{test_file}::{m.group(2)}", args.test_dir,
                     labels)
