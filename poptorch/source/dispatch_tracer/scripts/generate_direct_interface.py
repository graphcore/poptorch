# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import sys
import warnings

# PyTorch Schema types to C++ convertor.
schemaToCpp = {
    "int[]": "toIntVector",
    "int[1]": "toIntVector",
    "int[2]": "toIntVector",
    "int[3]": "toIntVector",
    "int[4]": "toIntVector",
    "int[6]": "toIntVector",
    "int[]?": "toOptionalIntVector",
    "int[1]?": "toOptionalIntVector",
    "int": "toInt",
    "int?": "toOptionalInt",
    "bool[]": "toBoolVector",
    "bool[1]": "toBoolVector",
    "bool[2]": "toBoolVector",
    "bool[3]": "toBoolVector",
    "bool": "toBool",
    "float": "toDouble",
    "float[]": "toFloatVector",
    "float[]?": "toOptionalFloatVector",
    "float?": "toOptionalDouble",
    "str": "toStr",
    "str?": "toOptionalStr",
    # We treat all scalars as double for now.
    "Scalar": "toDouble",
    "Scalar?": "toOptionalDouble",
    "ScalarType": "toCompilerType",
    "ScalarType?": "toOptionalCompilerType",
    'Tensor': "toTensor",
    'Tensor?': "toOptionalTensor",
    'Tensor[]': "toTensorVector",
    'Tensor?[]': "toTensorVector",
}


def addScope(string, scope='    '):
    return '\n'.join(s if s == '' else scope + s for s in string.split('\n'))


# Convert a tensor into it's constituent pieces.
# Given a tensor type from the yml like Tensor(a!) we turn that into a nicer format.
# Tensor(a!) -> tensor_id=a, is_inplace=True
# Tensor(a) -> tensor_id=a, is_view=True
# Tensor -> tensor_id=''
# Tensor(a!)[] -> tensor_id=a, is_list=True, is_inplace=True
# Tensor(a)[] ->  tensor_id=a, is_list=True, is_view=True
# Tensor[] -> tensor_id='', is_list=True
# If a tensor is not inplace or a view there is nothing to refer to in the context of arguments and returns.
# See the usage of these rules in native_functions.yml and in https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native for a full expansion of the precise rules (of which the above is an approximation).
class TensorInfo:
    def __init__(self, tensor):
        if not tensor.startswith("Tensor"):
            raise ValueError(f"Type {tensor} not implemented.")

        self.is_list = "[]" in tensor
        if self.is_list:
            tensor = tensor.replace("[]", "")

        self.tensor_id = ''
        self.is_inplace = False
        self.is_view = False

        # We have a normal non-aliasing tensor.
        if '(' not in tensor:
            if tensor != "Tensor":
                raise ValueError(f"Type {tensor} not implemented.")
            return

        self.is_inplace = '!' in tensor

        # If there are brackets but no `!` then this is view of a tensor.
        self.is_view = not self.is_inplace

        # The id of the tensor. This is the identifier given to map an input onto an output.
        self.tensor_id = tensor[len('Tensor('):-1]
        assert tensor[-1] == ")"

        # Remove the `!` if this is inplace.
        if self.is_inplace:
            assert self.tensor_id[-1] == '!'
            self.tensor_id = self.tensor_id[:-1]

    def add_output(self, index, named_tensors):
        # We will get a list of tensor IDs, which could be zero for optional
        # one ore more.
        outputs_code = f"""const auto &t_ids = mlir_output.at({str(index)}).tensor_ids;
auto requires_grad = requiresGrad(mlir_output.at({str(index)}).requires_grad_types, requires_grad_or);
"""
        if not self.is_list:
            outputs_code += "auto t_id = getSingleOptionalTensorId(t_ids);\n"

        # For each output tensor return it to pytorch in a different way
        # depending on what the schema tells us.
        if self.is_inplace:
            # Inplace operations should be inplaced versions of a certain input.
            if self.is_list:
                outputs_code += (
                    "stack.push_back(outputIsInplaceOfList(t_ids, "
                    f"{named_tensors[self.tensor_id]}_pytorch, "
                    "requires_grad));\n")
            else:
                outputs_code += ("stack.push_back(outputIsInplaceOf(t_id, "
                                 f"{named_tensors[self.tensor_id]}_pytorch, "
                                 "requires_grad.at(0)));\n")
        else:
            # Otherwise we are returning a new tensor or tensor list.
            if self.is_list:
                outputs_code += ("stack.push_back(makeEmptyOutputTensorList("
                                 "t_ids, requires_grad));\n")
            else:
                outputs_code += """if (t_id == poptorch_ir::none_id) {
    stack.push_back(makeEmptyOutputTensor(poptorch_ir::none_id, false));
} else {
    stack.push_back(makeEmptyOutputTensor(t_id, requires_grad.at(0)));
}
"""
        outputs_code = f'{{\n{addScope(outputs_code)}}}\n'
        return outputs_code


class ValueInfo:
    def __init__(self, value_schema):

        # Here we are processing the arguments from native functions.yml, i.e:
        # aten.contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)

        arg = value_schema.split('=')

        self.default = None
        if '=' in value_schema:
            self.default = arg[-1]

        # Remove default arguments.
        arg_name = arg[0].split(' ')[-1]

        # E.g Tensor(a) self -> name: self, type : Tensor(a)
        arg_name = arg_name.split(' ')[-1]
        self.type = value_schema.split(' ')[0]

        self.name = arg_name
        self.tensor_info = None
        self.is_tensor = 'Tensor' in self.type

        # Add the special tensor information. Some tensors don't have any, if there is a
        # `(` it means there is view or inplace information.
        # Tensor -> just a tensor.
        # Tensor(a) -> view of `a`
        # Tensor(a!) -> `a` modified inplace
        # Tensor(a!)[] -> list
        if self.is_tensor and '(' in self.type:
            self.tensor_info = TensorInfo(self.type)
            self.type = 'Tensor[]' if self.tensor_info.is_list else "Tensor"

    def assert_value_is_default(self,
                                stack_at_index,
                                aten_name,
                                default_value=None):
        if default_value is None:
            default_value = self.default

        if default_value == 'None':
            # If we know the expected values for the ignored arguments
            # emit checks for them
            return (f'ERROR_ON_MSG(!{stack_at_index}.isNone(), '
                    f'"{aten_name}: Poptorch does not handle {self.name}. '
                    'Expected it to be None");\n')

        if default_value in ('True', 'False'):
            return (
                f'ERROR_ON_MSG({stack_at_index}.toBool() != '
                f'{default_value.lower()}, "{aten_name}: Poptorch does not '
                f'handle {self.name}. Expected it to be {default_value}");\n')

        if default_value is None:
            warnings.warn(
                f'No default value for {self.name} in {aten_name}. You can '
                'use IgnoreArgs as a dictionary to provide a default '
                'value.')
        else:
            warnings.warn(
                f'Not implemented: default value ({default_value}) for '
                f'{self.name} in {aten_name} is not checked')

        return ''

    def get_argument(self, stack_at_index, aten_name):
        if self.type not in schemaToCpp:
            print(f"There is no c++ schema for {self.type} in {aten_name} "
                  f"from {__file__}.")
            print("You need to add one to schemaToCpp for compilation " +
                  "to succeed.")
            sys.exit(1)

        name = self.name + ('_pytorch' if 'Tensor' in self.type else '')

        return (f"[[maybe_unused]] auto {name} ="
                f" {schemaToCpp[self.type]}({stack_at_index});\n")

    def convert_to_tensor_id(self):
        if 'Tensor' in self.type:
            return (f'[[maybe_unused]] auto {self.name} = '
                    f'findTensor({self.name}_pytorch);\n')

        return ''

    def fill_requires_grad(self):
        tensor_name = f'{self.name}_pytorch'
        if 'Tensor[]' in self.type or 'Tensor?[]' in self.type:
            return f"""requires_grad_or |= std::any_of({tensor_name}.begin(), {tensor_name}.end(),
                                [this](const auto& t) {{ return t.requires_grad(); }});
"""
        if 'Tensor' in self.type:
            return f"""requires_grad_or |= {tensor_name}.requires_grad();
"""
        return ''


def add_op(function, parameters, outputs, named_tensors):
    return_type = "poptorch_ir::ODSTensorResults mlir_output =\n"
    return_type += "    "

    # Generate the call to the compiler function.
    function_decl = f"{return_type} _compiler."
    function_decl += function + "(" + ', '.join(parameters) + ");\n\n"

    # Clear the stack and add the outputs.
    function_decl += "// Pop pytorch inputs from stack\n"
    function_decl += "stack.clear();\n\n"

    # Handle each of the outputs.
    for index, output in enumerate(outputs):
        if output == "":  # `ie. -> ()`
            continue
        # Capture all metadata related to each of the output tensors.
        output_info = TensorInfo(output)

        function_decl += output_info.add_output(index, named_tensors)

    return function_decl


# Generate the c++ function which handles this operation.
def generate_cpp(op_target, canonicalised_args, outputs, named_tensors,
                 aten_name):
    # Some arguments we just completely ignore.
    args_to_ignore = {} if "IgnoreArgs" not in op_target else op_target[
        "IgnoreArgs"]

    # Some arguments are only marking the output tensor so we don't pass
    # them into the mlir call.
    key = "UnusedOutputArguments"
    unused_inplace_arg = {} if key not in op_target else op_target[key]

    function_decl = ""

    parameters = []

    for arg_index, arg in enumerate(canonicalised_args):
        stack_at_index = "stack.at(" + str(arg_index) + ")"

        # If the argument is in args_to_ignore we skip it
        if arg.name in args_to_ignore:
            # args_to_ignore is either a set or a dictionary. In the former
            # case we check the given value against the schema default value.
            # In the latter case we check the given value against the value in
            # the dictionary.
            #
            # Note: it appears that sets can sometimes be parsed as
            # dictionaries with None values by the yaml parser (we wish to
            # check against the schema in this case)
            is_in_dict = isinstance(
                args_to_ignore, dict) and args_to_ignore[arg.name] is not None
            arg_default = args_to_ignore[arg.name] if is_in_dict else None

            function_decl += arg.assert_value_is_default(
                stack_at_index, aten_name, arg_default)
            continue

        function_decl += arg.get_argument(stack_at_index, aten_name)

        if arg.name not in unused_inplace_arg:
            function_decl += arg.convert_to_tensor_id()
            function_decl += arg.fill_requires_grad()

            parameters.append(arg.name)

    function_decl = ("[[maybe_unused]] bool requires_grad_or = false;\n" +
                     function_decl)

    if "PopTorchDirect" in op_target:
        # We are dealing with a vanilla function.
        function_decl += add_op(op_target["PopTorchDirect"], parameters,
                                outputs, named_tensors)
    else:
        raise KeyError("Couldn't find a valid PopTorch direct mapping "
                       "(eg. PopTorchDirect)"
                       f" for {op_target}")

    return function_decl


class DirectMLIRGenerator:
    def __init__(self, header_file, cpp_file, lookup_file, namespace):

        # The files to output the results into.
        self.header = header_file
        self.cpp = cpp_file
        self.lookup = lookup_file
        self.namespace = namespace

    def gen_function(self, function_name, op_target, arguments, outputs):
        # We convert the args from the schema string to a list of lists.
        # The outer list being all arguments and the inner being the information
        # for that given argument. Most types are just [Name, Type] pairs in the list
        # however tensors can be views or inplace so we track that.
        canonicalised_args = []

        # Tensors which have been marked as being inplace/views will have an ID.
        named_tensors = {}
        is_view = False
        # Remove any optional value assignment (`=`) and take the last string which will be the name.
        for argument in arguments:
            argument = argument.strip()

            # The argument '*' is a special case for the python interface, we can ignore.
            if argument == "*":
                continue

            value = ValueInfo(argument)

            if value.tensor_info is not None:
                named_tensors[value.tensor_info.tensor_id] = value.name
                is_view |= value.tensor_info.is_view

            canonicalised_args.append(value)

        aten_name = function_name
        function_name = function_name.replace('.', '_')
        function_name = "{}_{}".format(self.namespace, function_name)

        # Generate the C++ impl.
        function_decl = "void MLIRDispatch::" + function_name
        function_decl += "(c10::Stack& stack) {\n"
        function_decl += addScope(
            generate_cpp(op_target, canonicalised_args, outputs, named_tensors,
                         aten_name))
        function_decl += "}\n"

        # Print the C++ impl.
        print(function_decl, file=self.cpp)

        # Generate the C++ header.
        print("void " + function_name + "(c10::Stack& stack);\n",
              file=self.header)

        # Generate the Aten Op to the C++ function map.
        print(
            f"{{\"{self.namespace}::{aten_name}\", [](MLIRDispatch& dispatch, "
            f"c10::Stack& stack) {{ dispatch.{function_name}(stack); }}}},",
            file=self.lookup)
