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
                    required=False,
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

parser.add_argument('--namespace',
                    required=True,
                    help='The symbol namespace in the supplied YAML file')

parse_args = parser.parse_args()

to_generate = yaml.safe_load(parse_args.ops_to_generate)

# Create a map of all the functions we are going to generate so we can look up the poptorch information for each native function.
ops_to_generate_dict = {}
for function in to_generate:
    ops_to_generate_dict[function['func']] = function

# Get the pytorch native functions
if parse_args.pytorch_base_native_function is not None:
    native_functions = yaml.safe_load(parse_args.pytorch_base_native_function)
    for function in native_functions:
        function_name = function['func'].split('(')[0]
        if function_name not in ops_to_generate_dict:
            continue
        ops_to_generate_dict[function_name]['native_func'] = function['func']

direct_gen = DirectMlirGenerator(parse_args.gen_hpp_file_path,
                                 parse_args.gen_cpp_file_path,
                                 parse_args.gen_lookup, parse_args.namespace)

for function_name, op_target in ops_to_generate_dict.items():

    # Split around the function return.
    signature, outputs = op_target['native_func'].split(' -> ')

    # Some operations return multiple outputs, e.g: (Tensor(a!) values, Tensor(b!) indices)
    # pylint: disable=literal-comparison
    is_multiple_outputs = outputs[0] is '('

    # If so convert into a list.
    if is_multiple_outputs:
        # Remove the `(` and `)`
        outputs = outputs[1:-1]

        # Split along the `,`
        tmp = outputs.split(', ')

        # Each output type.
        outputs = []

        # Currently we are only handling multiple tensor outputs, might need to change this in the future but prob not.
        for output in tmp:
            # Remove whitespace.
            output = output.split(' ')

            # Just add it if we don't need to look at the name.
            if len(output) == 1:
                outputs.append(output[0])
                continue

            # PyTorch sometimes gives them names, we only want the type it already contains an identifier.
            the_type = output[0]
            the_name = output[1]

            # Add it to the list
            outputs.append(the_type)
    else:
        # Otherwise we just have a single output, still add it to a list so the rest of the code can be cleaner.
        outputs = [outputs]

    # Remove the function name and the `(`/`)` argument brackets from the signature
    signature = signature[len(function_name) + 1:-1]

    if len(signature.strip()) == 0:
        arguments = []
    else:
        arguments = signature.split(',')

    # Parse the arguments and generate the C++ function
    direct_gen.gen_function(function_name, op_target, arguments, outputs)
