# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse

from schema_parser import SchemaParser
from generate_direct_interface import DirectMLIRGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert macro file to tablegen')

    parser.add_argument(
        '--ops-to-generate',
        required=True,
        type=argparse.FileType('r'),
        help='Generate a C++ binding for every call in this file.')

    # We use a separate native functions.yml file for the function signatures as
    # these may change and we don't want to have to write them out every time.
    parser.add_argument(
        '--pytorch-base-native-function',
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

    op_dict = SchemaParser(parse_args.ops_to_generate,
                           parse_args.pytorch_base_native_function)

    direct_gen = DirectMLIRGenerator(parse_args.gen_hpp_file_path,
                                     parse_args.gen_cpp_file_path,
                                     parse_args.gen_lookup,
                                     parse_args.namespace)

    def gen_function(function_name, op_target, arguments, kwargs, outputs):
        global direct_gen
        # Parse the arguments and generate the C++ function
        direct_gen.gen_function(function_name, op_target, arguments + kwargs,
                                outputs)

    op_dict.apply_to_dict(gen_function)
