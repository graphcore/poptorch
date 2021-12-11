# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse

import json

# This script from the JSON output of TableGen creates the following glue code:
# To create a C++ non mlir-API for PopTorch:
# * A Cpp header file which calls the MLIR builder with C++ types.
# * A Cpp impl for the above.

# And to help with PopTorch->MLIR.
# * Dispatch table which converts incoming PyTorch JIT IValue stuff
#   into C++ before calling the above functions.

parser = argparse.ArgumentParser(description='Convert macro file to tablegen')

parser.add_argument('--input-file',
                    required=True,
                    type=argparse.FileType('r'),
                    help='File to convert')

# Interface refers to the first category above. The functions called by PopTorch to build IR.
parser.add_argument(
    '--interface-header-file',
    required=True,
    type=argparse.FileType('w'),
    help='The header which creates the interface calls for PopTorch')

parser.add_argument('--interface-cpp-file',
                    required=True,
                    type=argparse.FileType('w'),
                    help='The cpp file which implements the interface calls')

# The dispatch for PyTorch JIT -> MLIR.
parser.add_argument('--jit-dispatch-table-output-file',
                    required=True,
                    type=argparse.FileType('w'),
                    help='File to output the jit dispatch table into')

parse_args = parser.parse_args()

active_op = None

ops = {}

# A targetable op looks like:
# class my_op : public ::mlir::Op<my_op, ... whole bunch of traits... ,  PoplarImplInterface::Trait> {
json_in = json.load(parse_args.input_file)

decl_types = {
    "F32ArrayAttr": "FLOAT_VEC",
    "StrArrayAttr": "STRING_VEC",
    "I32ArrayAttr": "INT_VEC",
    "I64ArrayAttr": "LONG_VEC",
    "I32Attr": "INT",
    "I64Attr": "LONG",
    "F32Attr": "FLOAT",
    "StrAttr": "STRING",
    "BoolAttr": "BOOL",
    "Poptorch_tensor": "TENSOR",
    "TypeAttr": "TYPE"
}

# Is a bit easier to work with if we convert the JSON into the old macro format even if we don't end up saving that to disk.
poptorch_ops = {}

for key in json_in.keys():
    if key.startswith("Poptorch_"):

        if "Poptorch_Op" not in json_in[key]["!superclasses"]:
            continue

        # Skip the stream copies. They have their own API.
        if "Poptorch_StreamCopy" in json_in[key]["!superclasses"]:
            continue

        op_name = json_in[key]["!name"].replace("Poptorch_", "")
        poptorch_ops[op_name] = {}

        args = ""
        body_args = ""
        poptorch_ops[op_name]["args"] = []

        poptorch_ops[op_name]["args_with_default_vals"] = {}

        # Convert the args into our little decl type system.
        for arg in json_in[key]["arguments"]["args"]:
            arg_name = arg[1]
            arg_type = arg[0]["def"]

            if arg_type not in decl_types:
                type_info = json_in[arg_type]

                # Variadic tensor.
                is_variadic_tensor = "Variadic" in type_info[
                    "!superclasses"] and type_info["baseType"][
                        "def"] == "Poptorch_tensor"
                if is_variadic_tensor:
                    arg_type = "TENSOR_VEC"

                # An attribute with a default value.
                is_default_valued = "DefaultValuedAttr" in type_info[
                    "!superclasses"] and type_info["baseAttr"][
                        "def"] in decl_types
                if is_default_valued:
                    arg_type = decl_types[type_info["baseAttr"]["def"]]

                    poptorch_ops[op_name]["args_with_default_vals"][
                        arg_name] = type_info["defaultValue"]

                # An optional operand.
                is_optional = "Optional" in type_info[
                    "!superclasses"] and type_info["baseType"][
                        "def"] in decl_types

                is_optional_attr = "OptionalAttr" in type_info[
                    "!superclasses"] and type_info["baseAttr"][
                        "def"] in decl_types

                if is_optional:
                    arg_type = decl_types[type_info["baseType"]["def"]]

                if is_optional_attr:
                    arg_type = "OPTIONAL_" + decl_types[type_info["baseAttr"]
                                                        ["def"]]

            else:
                arg_type = decl_types[arg_type]

            poptorch_ops[op_name]["args"].append((arg_name, arg_type))

            args += "ARG(" + arg_name + ", " + arg_type + ") COMMA "
            body_args += "BODY_ARG(" + arg_name + ") COMMA "

        # Remove the last comma
        args = args[:-len("COMMA ")]
        body_args = body_args[:-len("COMMA ")]

        if args == "":
            args = "NONE"
            body_args = "NONE"

        decl = "OP_DECL"

        poptorch_ops[op_name]["num_returns"] = len(
            json_in[key]["results"]["args"])

        # Mark no return ops.
        if len(json_in[key]["results"]["args"]) == 0:
            decl += "_NO_RETURN"
            poptorch_ops[op_name]["no_return"] = True

        decl += "( " + op_name + ", " + args + ", " + body_args + ")"

        # Print the generated decl.
        #print(decl, file=parse_args.decl_output_file)

# Create the builder Cpp/Hpp file.

builder_call_translations = {
    "FLOAT_VEC": "const std::vector<float> &",
    "STRING_VEC": "const std::vector<const char *> &",
    "INT_VEC": "const std::vector<std::int32_t> &",
    "LONG_VEC": "const std::vector<std::int64_t> &",
    "INT": "std::int32_t",
    "LONG": "std::int64_t",
    "OPTIONAL_LONG": "std::optional<std::int64_t>",
    "FLOAT": "float",
    "STRING": "const char *",
    "TENSOR": "poptorch_ir::TensorId",
    "TENSOR_VEC": "const std::vector<poptorch_ir::TensorId> &",
    "BOOL": "bool",
    "TYPE": "poptorch_ir::Type"
}

#"results": {"args": [
#    [{"def": "Poptorch_tensor", "kind": "def", "printable": "Poptorch_tensor"}, "result"],
#    [{"def": "Poptorch_tensor", "kind": "def", "printable": "Poptorch_tensor"}, "mean"],
#    [{"def": "Poptorch_tensor", "kind": "def", "printable": "Poptorch_tensor"}, "standard_deviation"]]

#define ARG(Name, Type) Type Name
#define BODY_ARG(Name) convert(Name, _impl->value_map)

for op_name in poptorch_ops:
    # Create a function with the same name as the compiler function to call.
    num_returns = poptorch_ops[op_name]["num_returns"]

    return_stmt = "" if num_returns == 0 else "return"

    if num_returns == 0:
        function_def = "void "
    elif num_returns == 1:
        function_def = "poptorch_ir::TensorId "
    else:
        function_def = "std::vector<poptorch_ir::TensorId> "

    function_def += "__OP_NAME_PLACEHOLDER__("

    # One space to give the comma removal thing something to remove when there are no args.
    func_args = " "
    parameters = ""

    # Turn the args into function signitures.
    for args in poptorch_ops[op_name]["args"]:
        name = args[0]
        arg_type = args[1]

        func_args += builder_call_translations[arg_type] + " " + name + " ,"

        # If the argument is an MLIR type we want to convert it first.
        # pylint: disable=literal-comparison
        if arg_type is "TYPE":
            parameters += ", _impl->convertType(" + name + ")"
        else:
            parameters += ", convert(" + name + ", _impl->value_map)"

    # Remove the last commas.
    func_args = func_args[:-1]

    cppFunction = function_def.replace("__OP_NAME_PLACEHOLDER__",
                                       "PoptorchCompiler::" + op_name)
    headerFunction = function_def.replace("__OP_NAME_PLACEHOLDER__", op_name)

    cppFunction += func_args + ") {\n"

    # Create the IR op.
    cppFunction += "auto tmp = _impl->builder.create<poptorch_ir::"
    cppFunction += op_name + ">(_impl->default_loc"
    cppFunction += parameters + ");\n"

    # Add the IR op to the graph.
    cppFunction += "_impl->main_graph.front().push_back(tmp);\n"

    # Map the output[s] into the IR map.
    if num_returns == 1:
        cppFunction += "_impl->value_map.push_back(tmp);"
        cppFunction += "return _impl->value_map.size() - 1;"
    elif num_returns > 1:
        # Create a temp vector for the new tensor IDs to return to the user
        cppFunction += "std::vector<poptorch_ir::TensorId> ids;\n"
        cppFunction += "ids.reserve(tmp.getNumResults());\n"

        cppFunction += "for(mlir::Value value : tmp.getResults()) {\n"
        cppFunction += "    _impl->value_map.push_back(value);\n"
        cppFunction += "   ids.push_back(_impl->value_map.size() - 1);\n"
        cppFunction += "}\n"

        cppFunction += "return ids;"

    # Add the function end scope.
    cppFunction += "}"

    # Print the header file
    print(headerFunction + func_args + ");",
          file=parse_args.interface_header_file)

    # Print the impl
    print(cppFunction, file=parse_args.interface_cpp_file)

disptach_cxx_cases = {
    "FLOAT_VEC": "const std::vector<float>&",
    "STRING_VEC": "const std::vector<const char*>&",
    "INT_VEC": "const std::vector<std::int32_t> &",
    "LONG_VEC": "const std::vector<std::int64_t> &",
    "INT": "std::int32_t",
    "LONG": "std::int64_t",
    "OPTIONAL_LONG": "std::optional<std::int64_t>",
    "FLOAT": "float",
    "STRING": "const char*",
    "BOOL": "bool",
    "TYPE": "poptorch_ir::Type"
}

# Generate the JIT dispatch table.
for op_name in poptorch_ops:
    # Create a function with the same name as the compiler function to call.
    num_returns = poptorch_ops[op_name]["num_returns"]

    return_stmt = "" if num_returns == 0 else "return"
    if num_returns == 0:
        function_def = "[[maybe_unused]] void "
    elif num_returns == 1:
        function_def = "[[maybe_unused]] poptorch_ir::TensorId "
    else:
        function_def = "[[maybe_unused]] std::vector<poptorch_ir::TensorId> "

    function_def += "JIT_" + op_name + "("
    function_def += "poptorch_ir::PoptorchCompiler &compiler,"
    function_def += "const std::vector<poptorch_ir::TensorId> &ids,"

    # Unpack all the arguments.
    unpack_args = "(void)ids;\n"
    parameters = " "

    if len(poptorch_ops[op_name]["args"]) > 0:
        unpack_args += "std::uint32_t index = 0; (void) index;\n"

    for args in poptorch_ops[op_name]["args"]:
        name = args[0]
        arg_type = args[1]

        # Get the default value of this argument if it exists.
        val = None
        if name in poptorch_ops[op_name]["args_with_default_vals"]:
            val = poptorch_ops[op_name]["args_with_default_vals"][name]

        if arg_type in disptach_cxx_cases:

            if val:
                function_def += disptach_cxx_cases[
                    arg_type] + " " + name + "=" + val + ","
            else:
                function_def += disptach_cxx_cases[arg_type] + " " + name + " ,"

            parameters += name + " ,"
        elif arg_type == "TENSOR":
            unpack_args += "poptorch_ir::TensorId " + name + "= ids[index++];\n"
            parameters += name + " ,"
        elif arg_type == "TENSOR_VEC":
            parameters += "ids ,"

    # Remove the last comma.
    parameters = parameters[:-1]
    function_def = function_def[:-1]

    function_def += ") {\n"

    print(function_def + unpack_args + return_stmt + " compiler." + op_name +
          "(" + parameters + ");\n }\n",
          file=parse_args.jit_dispatch_table_output_file)
