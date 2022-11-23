# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import sys
import warnings
import itertools
from typing import List
from schema_parser import ValueInfo, TypeInfo
from helpers import addScope

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
    "Tensor": "toTensor",
    "Tensor?": "toOptionalTensor",
    "Tensor[]": "toTensorVector",
    "Tensor?[]": "toTensorVector",
    "SymInt": "fromSymInt",
    "SymInt?": "fromOptionalSymInt",
    "SymInt[]": "fromSymIntList",
    "SymInt[0]": "fromSymIntList",
    "SymInt[1]": "fromSymIntList",
    "SymInt[2]": "fromSymIntList",
    "SymInt[3]": "fromSymIntList",
    "SymInt[4]": "fromSymIntList",
    "SymInt[5]": "fromSymIntList",
    "SymInt[6]": "fromSymIntList",
    "SymInt[7]": "fromSymIntList",
    "SymInt[8]": "fromSymIntList",
    "SymInt[9]": "fromSymIntList",
    "SymInt[]?": "fromOptionalSymIntList",
}

schemaToCppType = {
    "SymInt": "std::int64_t",
    "SymInt?": "std::optional<std::int64_t>",
    "SymInt[]": "std::vector<std::int64_t>",
    "SymInt[0]": "std::vector<std::int64_t>",
    "SymInt[1]": "std::vector<std::int64_t>",
    "SymInt[2]": "std::vector<std::int64_t>",
    "SymInt[3]": "std::vector<std::int64_t>",
    "SymInt[4]": "std::vector<std::int64_t>",
    "SymInt[5]": "std::vector<std::int64_t>",
    "SymInt[6]": "std::vector<std::int64_t>",
    "SymInt[7]": "std::vector<std::int64_t>",
    "SymInt[8]": "std::vector<std::int64_t>",
    "SymInt[9]": "std::vector<std::int64_t>",
    "SymInt[]?": "std::optional<std::vector<std::int64_t>>",
    "int[]": "std::vector<std::int64_t>",
    "int[1]": "std::vector<std::int64_t>",
    "int[2]": "std::vector<std::int64_t>",
    "int[3]": "std::vector<std::int64_t>",
    "int[4]": "std::vector<std::int64_t>",
    "int[6]": "std::vector<std::int64_t>",
    "int[]?": "std::optional<std::vector<std::int64_t>>",
    "int[1]?": "std::optional<std::vector<std::int64_t>>",
    "int": "std::int64_t",
    "int?": "std::optional<std::int64_t>",
    "bool[]": "std::vector<std::int64_t>",
    "bool[1]": "std::vector<std::int64_t>",
    "bool[2]": "std::vector<std::int64_t>",
    "bool[3]": "std::vector<std::int64_t>",
    "bool": "bool",
    "float": "double",
    "float[]": "std::vector<float>",
    "float[]?": "std::optional<std::vector<float>>",
    "float?": "std::optional<double>",
    # Note: strings aren't supported for now it's too tricky to store them on
    # the api boundary
    # "str": "std::vector<char>",
    # "str?": "std::optional<std::vector<char>>",
    # We treat all scalars as double for now.
    "Scalar": "double",
    "Scalar?": "std::optional<double>",
    "ScalarType": "poptorch_ir::Type",
    "ScalarType?": "std::optional<poptorch_ir::Type>",
    'Tensor': "std::shared_ptr<IpuTensorDetails>",
    'Tensor?': "std::shared_ptr<IpuTensorDetails>",
    'Tensor[]': "std::vector<std::shared_ptr<IpuTensorDetails>>",
    'Tensor?[]': "std::vector<std::shared_ptr<IpuTensorDetails>>",
}


def add_output(self: TypeInfo, index, named_tensors, args: List[ValueInfo],
               cpp_func: str):
    assert self.is_tensor

    # We will get a list of tensor IDs, which could be zero for optional
    # one ore more.
    outputs_code = f"""const auto &t_ids = mlir_output.at({str(index)}).tensor_ids;
auto requires_grad = requiresGrad(mlir_output.at({str(index)}).requires_grad_types, requires_grad_or);
"""
    if not self.is_list:
        outputs_code += "auto t_id = getSingleOptionalTensorId(t_ids);\n"

    output_view_var = 'nullptr'

    # If the output is a view we need to add the view information
    if self.is_view:
        class_name = cpp_func + 'TensorView'
        tensor_details_params = (f'getTensorDetails({x.name}_pytorch)'
                                 if x.type.is_tensor else x.name for x in args
                                 if not x.should_ignore)
        output_view = f'auto output_view = std::make_shared<{class_name}>('
        output_view += (',\n' + ' ' *
                        len(output_view)).join(tensor_details_params) + ");\n"

        if self.is_list:
            print(f"In {cpp_func} tensor lists cannot be views")
            #TODO(T71271): silently ignore for now.
            #sys.exit(1)

        outputs_code += output_view
        output_view_var = 'output_view'

    # For each output tensor return it to pytorch in a different way
    # depending on what the schema tells us.
    if self.is_inplace:
        # Inplace operations should be inplaced versions of a certain input.
        if self.is_list:
            outputs_code += ("stack.push_back(outputIsInplaceOfList(t_ids, "
                             f"{named_tensors[self.tensor_id]}_pytorch, "
                             f"requires_grad));\n")
        else:
            outputs_code += ("stack.push_back(outputIsInplaceOf(t_id, "
                             f"{named_tensors[self.tensor_id]}_pytorch, "
                             f"requires_grad.at(0), {output_view_var}));\n")
    else:
        # Otherwise we are returning a new tensor or tensor list.
        if self.is_list:
            outputs_code += ("stack.push_back(makeEmptyOutputTensorList("
                             "t_ids, requires_grad));\n")
        else:
            outputs_code += f"""if (t_id == poptorch_ir::none_id) {{
stack.push_back(makeEmptyOutputTensor(poptorch_ir::none_id, false, {output_view_var}));
}} else {{
stack.push_back(makeEmptyOutputTensor(t_id, requires_grad.at(0), {output_view_var}));
}}
"""
    outputs_code = f'{{\n{addScope(outputs_code)}}}\n'
    return outputs_code


def get_member_type(self: ValueInfo, aten_name):
    if self.type.str not in schemaToCppType:
        print(
            f"There is no c++ schema type for {self.type.str} in {aten_name} "
            f"from {__file__}.")
        print("You need to add one to schemaToCppType for compilation " +
              "to succeed.")
        sys.exit(1)
    return schemaToCppType[self.type.str]


def assert_value_is_default(self: ValueInfo, stack_at_index, aten_name):
    default_value = self.ignored_default

    if default_value is None:
        # If we know the expected values for the ignored arguments
        # emit checks for them
        return (f'ERROR_ON_MSG(!{stack_at_index}.isNone(), '
                f'"{aten_name}: Poptorch does not handle {self.name}. '
                'Expected it to be None");\n')

    if isinstance(default_value, bool):
        return (
            f'ERROR_ON_MSG({stack_at_index}.toBool() != '
            f'{str(default_value).lower()}, "{aten_name}: Poptorch does not '
            f'handle {self.name}. Expected it to be {default_value}");\n')

    if isinstance(default_value, str):
        return (f'ERROR_ON_MSG({stack_at_index}.toStringRef() != '
                f'"{default_value}",  "{aten_name}: Poptorch does not '
                f'handle {self.name}. Expected it to be {default_value}");\n')

    if isinstance(default_value, int):
        return (f'ERROR_ON_MSG({stack_at_index}.toInt() != '
                f'{default_value},  "{aten_name}: Poptorch does not '
                f'handle {self.name}. Expected it to be {default_value}");\n')

    if isinstance(default_value, float):
        return (f'ERROR_ON_MSG({stack_at_index}.toDouble() != '
                f'{default_value},  "{aten_name}: Poptorch does not '
                f'handle {self.name}. Expected it to be {default_value}");\n')

    warnings.warn(f'Not implemented: default value ({default_value}) for '
                  f'{self.name} in {aten_name} is not checked')

    return ''


def get_argument(self: ValueInfo, stack_at_index, aten_name):
    if self.type.str not in schemaToCpp:
        print(f"There is no c++ schema for {self.type.str} in {aten_name} "
              f"from {__file__}.")
        print("You need to add one to schemaToCpp for compilation " +
              "to succeed.")
        sys.exit(1)

    name = self.name + ('_pytorch' if 'Tensor' in self.type.str else '')

    return (f"[[maybe_unused]] auto {name} ="
            f" {schemaToCpp[self.type.str]}({stack_at_index});\n")


def convert_to_tensor_id(self: ValueInfo):
    if self.type.is_tensor:
        return (f'[[maybe_unused]] auto {self.name} = '
                f'findTensor({self.name}_pytorch);\n')

    return ''


def fill_requires_grad(self: ValueInfo):
    tensor_name = f'{self.name}_pytorch'
    if self.type.is_tensor:
        if self.type.is_list:
            return f"""requires_grad_or |= std::any_of({tensor_name}.begin(), {tensor_name}.end(),
                            [this](const auto& t) {{ return t.requires_grad(); }});
"""
        return f"""requires_grad_or |= {tensor_name}.requires_grad();
"""
    return ''


def add_op(cpp_func, args: List[ValueInfo], outputs, named_tensors):
    return_type = "poptorch_ir::ODSTensorResults mlir_output =\n"
    return_type += "    "

    # Generate the call to the compiler function.
    function_decl = f"{return_type} _compiler."
    function_decl += cpp_func + "(" + ', '.join(
        arg.name for arg in args if not arg.should_ignore) + ");\n\n"

    # Clear the stack and add the outputs.
    function_decl += "// Pop pytorch inputs from stack\n"
    function_decl += "stack.clear();\n\n"

    # Handle each of the outputs.
    for index, output in enumerate(outputs):
        # Note we are passing the same output view information to all the
        # outputs. This information is non-empty only when there is a single
        # output
        function_decl += add_output(output, index, named_tensors, args,
                                    cpp_func)

    return function_decl


# Generate the c++ function which handles this operation.
def generate_cpp(cpp_func, args: List[ValueInfo], outputs, named_tensors,
                 aten_name):
    function_decl = ""

    for arg_index, arg in enumerate(args):
        stack_at_index = "stack.at(" + str(arg_index) + ")"

        # If the argument is in args_to_ignore we skip it
        if arg.is_ignored:
            function_decl += assert_value_is_default(arg, stack_at_index,
                                                     aten_name)
            continue

        function_decl += get_argument(arg, stack_at_index, aten_name)

        if not arg.is_unused_output:
            function_decl += convert_to_tensor_id(arg)
            function_decl += fill_requires_grad(arg)

    function_decl = ("[[maybe_unused]] bool requires_grad_or = false;\n" +
                     function_decl)

    function_decl += add_op(cpp_func, args, outputs, named_tensors)

    return function_decl


def generate_view(function_name, args, outputs):
    if len(args) == 0:
        print(function_name + ": Views must have arguments")
        sys.exit(1)

    class_name = function_name + 'TensorView'
    members = map(lambda x: f'{get_member_type(x,function_name)} _{x.name}',
                  args)
    constructor_args = list(
        map(lambda x: f'{get_member_type(x,function_name)} {x.name}', args))
    initializers = map(lambda x: f'_{x.name}{{std::move({x.name})}}', args)

    def get_tensors_from_dispatch(arg):
        if arg.type.is_tensor:
            return f'auto {arg.name} = dispatch.ensureInDispatch(_{arg.name})'
        return f'auto &{arg.name} = _{arg.name};'

    find_tensors = map(get_tensors_from_dispatch, args)
    params = map(lambda x: x.name, args)

    if len(outputs) != 1:
        print("Views with multiple outputs aren't handled")
        sys.exit(1)

    if outputs[0].is_list:
        print(f"Views of list of {outputs[0].str} aren't handled")
        #TODO(T71271): silently ignore for now.
        #sys.exit(1)
    if not outputs[0].is_tensor:
        print(f"Views of {outputs[0].str} aren't handled")
        sys.exit(1)

    header = (f"friend class {class_name};\n"
              f"class {class_name} final : public ITensorView {{\n" +
              addScope(';\n'.join(members)) + f""";

public:
    explicit {class_name}({', '.join(constructor_args)});

    poptorch_ir::TensorId addViewToGraph(IDispatch &complier) override;
}};
""")

    cpp = (
        f"""MLIRDispatch::{class_name}::{class_name}({', '.join(constructor_args)})
    : {', '.join(initializers)} {{}}

poptorch_ir::TensorId MLIRDispatch::{class_name}::addViewToGraph(
    IDispatch &idispatch) {{
    auto& dispatch = reinterpret_cast<MLIRDispatch&>(idispatch);

""" + addScope(';\n'.join(find_tensors)) + f""";
    auto mlir_output = dispatch._compiler.{function_name}(
        {', '.join(params)});

    return mlir_output.at(0).tensor_ids.at(0);
}}
""")

    return header, cpp


class DirectMLIRGenerator:
    def __init__(self, header_file, cpp_file, lookup_file, namespace):

        # The files to output the results into.
        self.header = header_file
        self.cpp = cpp_file
        self.lookup = lookup_file
        self.namespace = namespace

    def gen_views(self, ops_to_generate_dict):
        def is_view(args):
            return any(value.type.is_view for value in args)

        # Find all of the python view ops
        view_functions = {
            function_name: schema
            for function_name, schema in ops_to_generate_dict.items()
            if is_view(schema[1])
        }

        views = {}

        # Find all the unique cpp views
        for python_name, (cpp_func, args, kwargs,
                          outputs) in view_functions.items():
            filtered_args = [
                arg for arg in itertools.chain(args, kwargs)
                if not arg.should_ignore
            ]
            # Check multiple references to the same cpp view have the same
            # arguments and outputs
            if cpp_func in views:
                if (filtered_args, outputs) != views[cpp_func]:
                    print('The arguments and outputs for the view '
                          f'{python_name} do not match the previously found '
                          f'values for {cpp_func}.')
            else:
                views[cpp_func] = (filtered_args, outputs)

        # Check that if a cpp function is marked as a view in one context it as
        # always marked as a view
        for python_name, (cpp_func, args, kwargs,
                          outputs) in ops_to_generate_dict.items():
            if (cpp_func in views
                    and not is_view(itertools.chain(args, kwargs))):
                print(f'{cpp_func} was previously used as a view function '
                      'whereas it is being used without view semantics in '
                      f'{python_name}.')

        header = ''
        cpp = ''
        for cpp_func, (args, outputs) in views.items():
            temp_header, temp_cpp = generate_view(cpp_func, args, outputs)
            header += temp_header
            cpp += temp_cpp

        # Print the C++ impl.
        print(cpp, file=self.cpp)

        # Generate the C++ header.
        print(header, file=self.header)

    def gen_function(self, function_name, cpp_func, arguments, outputs):
        # Tensors which have been marked as being inplace/views will have an ID.
        named_tensors = {
            value.type.tensor_id: value.name
            for value in arguments if value.type.tensor_id != ''
        }

        header = ''
        cpp = ''

        aten_name = function_name
        function_name = function_name.replace('.', '_')
        function_name = "{}_{}".format(self.namespace, function_name)

        header += f'void {function_name}(c10::Stack& stack);\n'

        # Generate the C++ impl.
        cpp += "void MLIRDispatch::" + function_name
        cpp += "(c10::Stack& stack) {\n"
        cpp += addScope(
            generate_cpp(cpp_func, arguments, outputs, named_tensors,
                         aten_name))
        cpp += "}\n"

        # Print the C++ impl.
        print(cpp, file=self.cpp)

        # Generate the C++ header.
        print(header, file=self.header)

        # Generate the Aten Op to the C++ function map.
        print(
            f"{{\"{self.namespace}::{aten_name}\", [](MLIRDispatch& dispatch, "
            f"c10::Stack& stack) {{ dispatch.{function_name}(stack); }}}},",
            file=self.lookup)
