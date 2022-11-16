# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import ast
import itertools
import re
import sys
import yaml


# Parse a tensor type string into it's constituent pieces.
# Given a tensor type from the yml like Tensor(a!) we turn that into a nicer format.
# Tensor(a!) -> tensor_id=a, is_inplace=True
# Tensor(a) -> tensor_id=a, is_view=True
# Tensor(a -> *) -> tensor_id=a, is_view=True (output may use `a` in a list)
# Tensor -> tensor_id=''
# Tensor(a!)[] -> tensor_id=a, is_list=True, is_inplace=True
# Tensor(a)[] ->  tensor_id=a, is_list=True, is_view=True
# Tensor[] -> tensor_id='', is_list=True
# If a tensor is not inplace or a view there is nothing to refer to in the context of arguments and returns.
# See the usage of these rules in native_functions.yml and in https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native for a full expansion of the precise rules (of which the above is an approximation).
class TypeInfo:
    def __init__(self, type_str):
        self.str = re.sub(r'\(.*\)', '', type_str)
        self.is_tensor = 'Tensor' in type_str

        list_match = re.search(r'\[(\d*?)\]', type_str)

        def to_int(str):
            if str == '':
                return None
            return int(str)

        self.is_list = bool(list_match)
        self.num_elements = to_int(list_match.groups()[0]) if list_match else 1

        self.is_optional = '?' in type_str
        self.base_type = re.match(r'\w+', type_str).group(0)

        self.is_inplace = False
        self.is_view = False
        self.tensor_id = ''

        if '(' not in type_str:
            return

        if not self.is_tensor:
            raise f'Cannot handle non tensor inplace output: {type_str}'

        self.is_inplace = '!' in type_str

        # If there are brackets but no `!` then this is view of a tensor.
        self.is_view = not self.is_inplace

        # The id of the tensor. This is the identifier given to map an input onto an output.
        self.tensor_id = re.search(r"Tensor\(([a-z]+)(| .*|!)\)",
                                   type_str).group(1)
        assert type_str[-1] in [")", "]"]


class ValueInfo:
    def __init__(self, value_schema, args_to_ignore, unused_inplace_args,
                 aten_name):
        # Here we are processing the arguments from native functions.yml, i.e:
        # aten.contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)

        arg = value_schema.split('=')

        has_default = '=' in value_schema
        self.default = None
        if has_default:
            if arg[-1] == 'Mean':
                self.default = 0
            else:
                self.default = ast.literal_eval(arg[-1])

        # Remove default arguments.
        arg_name = arg[0].split(' ')[-1]

        # E.g Tensor(a) self -> name: self, type : Tensor(a)
        arg_name = arg_name.split(' ')[-1]
        type_str = value_schema.rsplit(' ', 1)[0]

        self.name = arg_name
        self.type = TypeInfo(type_str)

        self.is_unused_output = self.name in unused_inplace_args
        self.is_ignored = self.name in args_to_ignore
        self.ignored_default = None

        if self.is_ignored:
            assert not self.is_unused_output
            # args_to_ignore is either a set or a dictionary. In the former
            # case we check the given value against the schema default value.
            # In the latter case we check the given value against the value in
            # the dictionary.
            #
            # Note: it appears that sets can sometimes be parsed as
            # dictionaries with None values by the yaml parser (we wish to
            # check against the schema in this case)
            if (isinstance(args_to_ignore, dict)
                    and args_to_ignore[self.name] is not None):
                self.ignored_default = ast.literal_eval(
                    args_to_ignore[self.name])
            elif has_default:
                self.ignored_default = self.default
            else:
                print(f'No default value for {self.name} in {aten_name}. You '
                      'may use IgnoreArgs as a dictionary to provide a '
                      'default value.')
                sys.exit(1)

    @property
    def should_ignore(self):
        return self.is_ignored or self.is_unused_output


def _process_schema(function_name, op_target):
    # Split around the function return.

    # In some cases there are more than 1 '->', make sure to match the one before the outputs.
    # e.g split_with_sizes(Tensor(a -> *) self, int[] split_sizes, int dim=0) -> Tensor(a)[]
    signature, outputs = op_target['native_func'].rsplit(' -> ', 1)

    # Some operations return multiple outputs, e.g: (Tensor(a!) values, Tensor(b!) indices)
    is_multiple_outputs = outputs.startswith('(')

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

    arguments = [argument.strip() for argument in arguments]

    args_to_ignore = ({} if "IgnoreArgs" not in op_target else
                      op_target["IgnoreArgs"])

    # Some arguments are only marking the output tensor so we don't pass
    # them into the mlir call.
    unused_inplace_args = ({} if "UnusedOutputArguments" not in op_target else
                           op_target["UnusedOutputArguments"])

    # Get metadata about each argument
    # Note: the argument '*' signifies the start of the keyword arguments
    # and can otherwise be ignored
    spl_arguments = [
        list(y) for x, y in itertools.groupby(arguments, lambda x: x == '*')
        if not x
    ]
    assert len(spl_arguments) <= 2

    arguments = [
        ValueInfo(argument, args_to_ignore, unused_inplace_args, function_name)
        for argument in spl_arguments[0]
    ] if len(spl_arguments) > 0 else []
    kwargs = [
        ValueInfo(argument, args_to_ignore, unused_inplace_args, function_name)
        for argument in spl_arguments[1]
    ] if len(spl_arguments) > 1 else []

    outputs = [TypeInfo(output) for output in outputs if output != '']

    if "PopTorchDirect" not in op_target:
        raise KeyError("Couldn't find a valid PopTorch direct mapping "
                       f"(eg. PopTorchDirect) for {op_target}")

    return op_target["PopTorchDirect"], arguments, kwargs, outputs


class SchemaParser:
    def __init__(self, ops_to_generate, schema_file=None):
        to_generate = yaml.safe_load(ops_to_generate)

        # Create a map of all the functions we are going to generate so we can look up the poptorch information for each native function.
        ops_to_generate_dict = {}
        for function in to_generate:
            ops_to_generate_dict[function['func']] = function

        # Get the pytorch native functions
        if schema_file is not None:
            native_functions = yaml.safe_load(schema_file)
            for function in native_functions:
                function_name = function['func'].split('(')[0]
                if function_name not in ops_to_generate_dict:
                    continue
                ops_to_generate_dict[function_name]['native_func'] = function[
                    'func']

        self.ops_to_generate_dict = {
            function_name: _process_schema(function_name, op_target)
            for function_name, op_target in ops_to_generate_dict.items()
        }

    def apply_to_dict(self, func):
        for function_name, schema in self.ops_to_generate_dict.items():
            # Parse the arguments and apply
            func(function_name, *schema)
