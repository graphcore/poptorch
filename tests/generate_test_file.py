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
parser.add_argument("--add-to-sys-path", help="Path to add to sys.path")
parser.add_argument("--extra-pytest-args",
                    type=str,
                    help=("Extra arguments to pass to pytest when generating "
                          "the list of tests."))

args = parser.parse_args()

if args.add_to_sys_path:
    for path in args.add_to_sys_path.split(";"):
        print(f"Adding {path}")
        sys.path.insert(0, path)

# This script doesn't actually need poptorch, but pytest later on will import
# it while compiling the list of tests and if it fails then we usually don't
# get the reason (Because the collection happens in a subprocess).
import poptorch  # pylint: disable=unused-import,wrong-import-position

# Collect the list of tests:
list_tests = io.StringIO()
pytest_args = ["-x", args.test_dir, "--collect-only", "-q"]
extra_args = []
if args.extra_pytest_args:
    arg = args.extra_pytest_args.replace("\"", "")
    if arg:
        extra_args = arg.split(",")
        pytest_args += extra_args

with contextlib.redirect_stdout(list_tests):
    retval = pytest.main(pytest_args)

assert retval == pytest.ExitCode.OK, f"{str(retval)}: {list_tests.getvalue()}"

# Run all the tests contained in these files in a single process
# because they're small / short to run (Under 1 minute)
# NB tests requring custom_ops libraries must go in here
#pylint: disable=line-too-long
# yapf: disable
short_tests = [
    "activations_test.py",
    "batching_test.py",
    "blas_test.py",
    "buffers_test.py",
    "custom_loss_test.py",
    "custom_ops_attributes_test.py",
    "custom_ops_test.py",
    "inputs_test.py",
    "loop_test.py",
    "lstm_test.py",
    "non_contiguous_tensors_test.py",
    "ops_test.py",
    "options_test.py",
    "outputs_test.py",
    "pipelining_test.py",
    "poplar_executor_test.py",
    "precompilation_test.py",
    "random_sampling_test.py",
    "replicated_graph_test.py",
    "requires_grad_test.py",
    "sharding_test.py",
]

# The only tests that should be run in doc-only builds.
docs_only_tests = [
        "test_doc_urls.py"
]

long_tests = [
    "bert_small_and_medium_test.py::test_bert_medium_result",
    "fine_tuning_test.py",
    "half_test.py::test_resnet",
    "io_performance_test.py::test_compare_io_performance",
    "torch_nn_test.py::test_pytorch_nn[trace_model:False-use_half:False-test_name:test_nn_Conv2d_circular_stride2_pad2]",
    "torch_nn_test.py::test_pytorch_nn[trace_model:True-use_half:False-test_name:test_nn_Conv2d_circular_stride2_pad2]",
    "torchvision_inference_test.py::test_googlenet",
    "torchvision_inference_test.py::test_inception_v3",
    "torchvision_inference_test.py::test_mnasnet1_0",
    "torchvision_inference_test.py::test_mobilenet_v2",
    "torchvision_inference_test.py::test_resnet18",
    "torchvision_inference_test.py::test_resnext50_32x4d",
    "torchvision_inference_test.py::test_squeezenet1_1",
]

# Tests depending on external data being downloaded to run.
external_data_tests = [
    "bert_small_and_medium_test.py::test_bert_medium_result",
    "bert_small_and_medium_test.py::test_bert_small",
    "bert_small_and_medium_test.py::test_bert_small_half",
]
# yapf: enable

# Tests that cannot run in parallel with other tests
serial_tests = [
    "attach_detach_test.py",
    "attach_detach_wait_for_ipu_test.py",
    "io_performance_test.py::test_compare_io_performance",
]
#pylint: enable=line-too-long


def add_test(output, test, root_folder, folder, test_id, test_properties,
             extra_args):
    extra = " ".join([f"\"{a}\"" for a in extra_args])
    output.write(
        f"add_test({test} \"{root_folder}/timeout_handler.py\" \"python3\""
        f" \"-m\" \"pytest\" \"-sv\" \"{folder}/{test}\" "
        f"\"--junitxml=junit/junit-test{test_id}.xml\" {extra})\n")

    props_string = " ".join(f"{k} {v}" for k, v in test_properties.items())

    output.write(f"set_tests_properties({test} PROPERTIES\n{props_string})\n")


work_dir = os.getcwd()

with open(args.output_file, "w") as output:
    test_id = 0
    # Add the short_tests files
    for test in short_tests:
        add_test(output, test, args.test_dir, args.test_dir, test_id, {
            "LABELS": "short",
            "WORKING_DIRECTORY": work_dir
        }, extra_args)
        test_id += 1

    # Process the list of tests returned by pytest
    for test in list_tests.getvalue().split("\n"):
        # Extract the file name from the test name
        m = re.match("^(.*)::(.*)", test)
        if m:
            test_properties = {"WORKING_DIRECTORY": work_dir}
            # Mark tests as timed out 1 second after TEST_TIMEOUT appears in
            # their output (see tests/timeout_handler.py)
            test_properties["TIMEOUT_AFTER_MATCH"] = "\"1;TEST_TIMEOUT\""
            # Use os.path.basename() to ensure we only have
            # the filename
            test_file = os.path.basename(m.group(1))

            dir_path = args.test_dir

            if os.path.dirname(m.group(1)) != "tests":
                # Convert to a proper path.
                path = os.path.normpath(m.group(1))

                # Seperate out the dirs and remove the "tests" from the start
                # and the test name from the end.
                separate_dirs = path.split(os.sep)[1:-1]

                # Append the dirs to the start of the root dir one.
                dir_path = os.path.join(dir_path, *separate_dirs)

            if test_file in short_tests:
                continue
            labels = []
            if test in long_tests:
                labels.append("long")
            if test in external_data_tests:
                labels.append("external_data")
            if test_file in docs_only_tests:
                labels.append("docs_only")

            if test_file in serial_tests:
                test_properties['RUN_SERIAL'] = 'TRUE'

            if labels:
                test_properties['LABELS'] = ";".join(labels)

            add_test(output, f"{test_file}::{m.group(2)}", args.test_dir,
                     dir_path, test_id, test_properties, extra_args)
            test_id += 1
