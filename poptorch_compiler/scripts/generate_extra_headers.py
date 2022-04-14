# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import sys

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

parse_args = parser.parse_args()

active_op = None

ops = {}

# A targetable op looks like:
# class my_op : public ::mlir::Op<my_op, ... whole bunch of traits... ,  PoplarImplInterface::Trait> {
json_in = json.load(parse_args.input_file)

attr_types = {
    "F32ArrayAttr": "FLOAT_VEC",
    "StrArrayAttr": "STRING_VEC",
    "I32ArrayAttr": "INT_VEC",
    "I64ArrayAttr": "LONG_VEC",
    "BoolArrayAttr": "BOOL_VEC",
    "I32Attr": "INT",
    "I64Attr": "LONG",
    "F32Attr": "FLOAT",
    "F64Attr": "DOUBLE",
    "StrAttr": "STRING",
    "BoolAttr": "BOOL",
    "TypeAttr": "TYPE"
}

tensor_types = {
    "Poptorch_tensor": "TENSOR",
}

# Is a bit easier to work with if we convert the JSON into the old macro format even if we don't end up saving that to disk.
poptorch_ops = {}


class ValueType:
    def __init__(self, name, type_str, default_value=None):
        self._name = name
        self._type_str = type_str
        self._default_value = default_value

    @property
    def default_value(self):
        return self._default_value

    @property
    def name(self):
        return self._name

    def macro_type(self):
        """ Returns the type used in C++ macros i.e. where ARG(Type, Name) is
        used"""
        try:
            return attr_types[self._type_str]
        except KeyError:
            return tensor_types[self._type_str]


class OptionalValueType(ValueType):
    def __init__(self, name, type_str):
        super().__init__(name, type_str)

    def macro_type(self):
        return "OPTIONAL_" + super().macro_type()


class VariadicValueType(ValueType):
    def __init__(self, name, type_str):
        super().__init__(name, type_str)

    def macro_type(self):
        if self._type_str not in tensor_types:
            raise ValueError("Only tensor types can be variadic")
        return super().macro_type() + "_VEC"


def vals_from_json(json_in, op_key, sub_key, allow_attributes=True):
    assert any([
        x in json_in[op_key]["!superclasses"]
        for x in ['Poptorch_Op', 'Poptorch_NotImplementedOp']
    ])

    arg_types = []

    for arg in json_in[op_key][sub_key]["args"]:
        arg_name = arg[1]
        arg_type_str = arg[0]["def"]

        # Basic attribute/tensor types can be handled simply
        if arg_type_str in attr_types or arg_type_str in tensor_types:
            arg_types.append(ValueType(arg_name, arg_type_str))
            continue

        # Otherwise the type should be anonymous type whose traits can be
        # looked up, or "Poptorch_tensorlist" which is an instance of
        # Variadic<Poptorch_tensor>
        if ("anonymous" not in arg_type_str
                and arg_type_str != "Poptorch_tensorlist"):
            raise ValueError(f"{arg_type_str} unknown and not anonymous.")

        resolved_type_details = json_in[arg_type_str]
        resolved_type_superclasses = resolved_type_details['!superclasses']

        # Many types have Constraint or Attr/TypeContraint
        if 'Constraint' in resolved_type_superclasses:
            resolved_type_superclasses.remove('Constraint')
        if 'AttrConstraint' in resolved_type_superclasses:
            resolved_type_superclasses.remove('AttrConstraint')
        if 'TypeConstraint' in resolved_type_superclasses:
            resolved_type_superclasses.remove('TypeConstraint')

        # Sometimes DefaultValuedAttr/OptionalAttr come alone and
        # sometimes with Attr
        is_attr = ("Attr" in resolved_type_superclasses
                   or "DefaultValuedAttr" in resolved_type_superclasses
                   or "OptionalAttr" in resolved_type_superclasses)

        if is_attr:
            if not allow_attributes:
                raise ValueError(f"Attributes not permitted for {sub_key}")

            if "Attr" in resolved_type_superclasses:
                resolved_type_superclasses.remove("Attr")

            type_str = resolved_type_details["baseAttr"]["def"]
            if type_str not in attr_types:
                print(f"Unhandled type {type_str} in {__file__}")

            if len(resolved_type_superclasses) != 1:
                print(f"{arg_name} has an unexpected number of superclasses.")
                sys.exit(1)

            single_attribute = resolved_type_superclasses[0]
            if single_attribute == "OptionalAttr":
                arg_types.append(OptionalValueType(arg_name, type_str))
            elif single_attribute == "DefaultValuedAttr":
                arg_types.append(
                    ValueType(arg_name, type_str,
                              resolved_type_details["defaultValue"]))
            else:
                print(
                    f"Attribute {single_attribute} not handled in {__file__}")
                sys.exit(1)

            continue

        # Handle tensors
        if len(resolved_type_superclasses) != 1:
            print(f"{arg_name} has an unexpected number of superclasses.")
            sys.exit(1)

        # Only attribute sshould have a default value
        assert "default_value" not in resolved_type_details

        type_str = resolved_type_details["baseType"]["def"]
        if type_str not in tensor_types:
            print(f"Unhandled type {type_str} in {__file__}")
            sys.exit(1)

        single_attribute = resolved_type_superclasses[0]

        if single_attribute == "Optional":
            arg_types.append(OptionalValueType(arg_name, type_str))
        elif single_attribute == "Variadic":
            arg_types.append(VariadicValueType(arg_name, type_str))
        else:
            print(f"Attribute {single_attribute} not handled in {__file__}")
            sys.exit(1)

    return arg_types


# Convert all the arguments from json to ValueType
def args_from_json(json_in, op_key):
    return vals_from_json(json_in, op_key, "arguments")


def returns_from_json(json_in, op_key):
    return vals_from_json(json_in, op_key, "results", False)


for key in json_in.keys():
    if key.startswith("Poptorch_"):
        if all(x not in json_in[key]["!superclasses"]
               for x in ["Poptorch_Op", "Poptorch_NotImplementedOp"]):
            continue

        # Skip the stream copies. They have their own API.
        if "Poptorch_StreamCopy" in json_in[key]["!superclasses"]:
            continue

        op_name = json_in[key]["!name"].replace("Poptorch_", "")
        poptorch_ops[op_name] = {}

        # Convert the args into our little decl type system.
        poptorch_ops[op_name]["args"] = args_from_json(json_in, key)
        poptorch_ops[op_name]["returns"] = returns_from_json(json_in, key)

# Create the builder Cpp/Hpp file.
builder_call_translations = {
    "FLOAT_VEC": "const std::vector<float> &",
    "STRING_VEC": "const std::vector<const char *> &",
    "INT_VEC": "const std::vector<std::int32_t> &",
    "LONG_VEC": "const std::vector<std::int64_t> &",
    "BOOL_VEC": "const std::vector<std::int64_t> &",  # Avoid packing issue
    "INT": "std::int32_t",
    "LONG": "std::int64_t",
    "OPTIONAL_LONG": "std::optional<std::int64_t>",
    "FLOAT": "float",
    "DOUBLE": "double",
    "OPTIONAL_DOUBLE": "std::optional<double>",
    "STRING": "const char *",
    "TENSOR": "poptorch_ir::TensorId",
    "OPTIONAL_TENSOR": "poptorch_ir::OptionalTensorId",
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
    #print(op_name)

    # Create a function with the same name as the compiler function to call.
    # Allow for each return to be optional, normal or variadic.
    # This is a vector of vecor of TensorIds.
    function_def = "poptorch_ir::ODSTensorResults "

    # if num_returns == 0:
    #     function_def = "void "
    # elif num_returns == 1:
    #     function_def = "poptorch_ir::TensorId "
    # else:
    #     function_def = "std::vector<poptorch_ir::TensorId> "

    function_def += "__OP_NAME_PLACEHOLDER__("

    # One space to give the comma removal thing something to remove when there are no args.
    func_args = []
    parameters = []

    # Turn the args into function signitures.
    for arg in poptorch_ops[op_name]["args"]:
        func_args.append(builder_call_translations[arg.macro_type()] + " " +
                         arg.name)

        # If the argument is an MLIR type we want to convert it first.
        # pylint: disable=literal-comparison
        if arg.macro_type() is "TYPE":
            parameters.append("_impl->convertType(" + arg.name + ")")
        else:
            parameters.append("convert(" + arg.name + ", _impl->value_map)")

    func_args_str = ", ".join(func_args)

    cppFunction = function_def.replace("__OP_NAME_PLACEHOLDER__",
                                       "PoptorchCompiler::" + op_name)
    headerFunction = function_def.replace("__OP_NAME_PLACEHOLDER__", op_name)

    cppFunction += func_args_str + ") {\n\n"

    # Create the IR op.
    cppFunction += "auto tmp = _impl->createOp<poptorch_ir::"
    cppFunction += op_name + ">(poptorch_ir::AddToGraph::MAIN_GRAPH, "
    cppFunction += ", ".join(parameters) + ");\n\n"

    # Allow for each return to be optional, normal or variadic.
    cppFunction += "poptorch_ir::ODSTensorResults results;\n"

    for arg_num, arg in enumerate(poptorch_ops[op_name]["returns"]):
        cppFunction += "results.emplace_back();\n"
        cppFunction += "// Each result may be optional or variadic.\n"
        cppFunction += "for(mlir::Value value : tmp.getODSResults("
        cppFunction += str(arg_num) + ")) {\n"
        cppFunction += "  _impl->value_map.push_back(value);\n"
        cppFunction += "  results.back()."
        cppFunction += "push_back(_impl->value_map.size() - 1);\n"
        cppFunction += "}\n\n"

    cppFunction += "return results;\n"

    # Add the function end scope.
    cppFunction += "}\n\n"

    # Save the header file and implementation
    print(headerFunction + func_args_str + ");",
          file=parse_args.interface_header_file)
    print(cppFunction, file=parse_args.interface_cpp_file)
