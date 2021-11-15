# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import yaml

from generate_direct_interface import DirectMlirGenerator

parser = argparse.ArgumentParser(description='Convert macro file to tablegen')

parser.add_argument('--ops-to-generate',
                    required=True,
                    type=argparse.FileType('r'),
                    help='Generate a C++ binding for every call in this file.')

# We use a seperate native functions.yml file for the function signitures as
# these may change and we don't want to have to write them out every time.
parser.add_argument('--pytorch-base-native-function',
                    required=True,
                    type=argparse.FileType('r'),
                    help='Native_functions.yml for this version of PyTorch.')

parser.add_argument('--gen-cpp-file-path',
                    required=True,
                    type=argparse.FileType('w'),
                    help='Output c++ impl file')

parser.add_argument('--gen-hpp-file-path',
                    required=True,
                    type=argparse.FileType('w'),
                    help='Output header file')

parser.add_argument('--gen-lookup',
                    required=True,
                    type=argparse.FileType('w'),
                    help='The symbol to function lookup map')

parse_args = parser.parse_args()

to_generate = yaml.safe_load(parse_args.ops_to_generate)

# Create a map of all the functions we are going to generate so we can look up the poptorch information for each native function.
ops_to_generate_dict = {}
for function in to_generate:
    ops_to_generate_dict[function['func']] = function

# Get the pytorch native functions
native_functions = yaml.safe_load(parse_args.pytorch_base_native_function)

direct_gen = DirectMlirGenerator(parse_args.gen_hpp_file_path,
                                 parse_args.gen_cpp_file_path,
                                 parse_args.gen_lookup)

for function in native_functions:

    function_name = function['func'].split('(')[0]

    if function_name in ops_to_generate_dict:

        # The poptorch target i.e the yml entry we have added for this class.
        op_target = ops_to_generate_dict[function_name]

        # Split around the function return.
        signature, outputs = function['func'].split(' -> ')

        # Remove the function name and the `(`/`)` argument brackets from the signature
        signature = signature[len(function_name) + 1:-1]

        arguments = signature.split(',')

        # Parse the arguments and generate the C++ function
        direct_gen.gen_function(function_name, op_target, arguments, outputs)
