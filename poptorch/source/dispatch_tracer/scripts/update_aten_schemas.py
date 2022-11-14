# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import argparse

# Use ruamel.yaml in order to preserve comments in the updated file
import ruamel.yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert macro file to tablegen')

    parser.add_argument(
        '--ops-to-generate',
        required=True,
        type=str,
        help='Update the schemas for all the functions in that file.')

    # We use a separate native functions.yml file for the function signatures as
    # these may change and we don't want to have to write them out every time.
    parser.add_argument(
        '--pytorch-base-native-function',
        required=False,
        type=argparse.FileType('r'),
        help='Native_functions.yml for this version of PyTorch.')

    parse_args = parser.parse_args()

    yaml = ruamel.yaml.YAML()
    yaml.width = 1000  # prevent line wrap
    with open(parse_args.ops_to_generate, "r") as f:
        ops = yaml.load(f)

    native_functions = yaml.load(parse_args.pytorch_base_native_function)
    # Index the function signatures by function name i.e:
    # { "fn_name" : "fn_name(arg, ...) -> returnType"}
    functions = {
        fn.split('(')[0]: fn
        for fn in [f['func'] for f in native_functions]
    }

    for op in ops:
        name = op['func']
        signature = functions.get(name)
        assert signature, (f"Can't find {name} in " +
                           parse_args.pytorch_base_native_function)
        op["native_func"] = signature

    with open(parse_args.ops_to_generate, "w") as f:
        ops = yaml.dump(ops, f)
