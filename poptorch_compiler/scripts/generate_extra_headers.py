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
    "F64ArrayAttr": "DOUBLE_VEC",
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
    "Poptorch_tensor": ("TENSOR", "OR_INPUTS"),
    "Poptorch_float_tensor": ("TENSOR", "OR_INPUTS"),
    "Poptorch_integral_tensor": ("TENSOR", "OR_INPUTS"),
    "Poptorch_non_boolean_tensor": ("TENSOR", "OR_INPUTS"),
    "Poptorch_tensor_no_grad": ("TENSOR", "FALSE"),
}

# Is a bit easier to work with if we convert the JSON into the old macro format even if we don't end up saving that to disk.
poptorch_ops = {}


class ValueType:
    def __init__(self, name, type_str, default_value=None):
        self._name = name
        self._type_str = type_str
        self._default_value = default_value

    def updateDefaultValue(self, default_value):
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
            return tensor_types[self._type_str][0]

    def requires_grad_type(self):
        if self._type_str not in tensor_types:
            return None
        return tensor_types.get(self._type_str)[1]


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


def resolve_value(json_in, arg_name, type_str, is_attr=False):
    """ For a return value or argument in the json extract the type, default
        value and whether the value is an attribute
    """
    # Basic attribute/tensor types can be handled simply
    if type_str in attr_types or type_str in tensor_types:
        return (ValueType(arg_name, type_str), False)

    # Otherwise the type should be an anonymous type whose traits can be looked
    # up, or "Poptorch_tensorlist" which is an instance of
    # Variadic<Poptorch_tensor>
    if ("anonymous" not in type_str and type_str not in [
            "Poptorch_tensorlist", "Poptorch_tensorlist_no_grad"
    ]):
        raise ValueError(f"{type_str} unknown and not anonymous.")

    resolved_type_details = json_in[type_str]
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
    is_attr |= ("Attr" in resolved_type_superclasses
                or "DefaultValuedAttr" in resolved_type_superclasses
                or "OptionalAttr" in resolved_type_superclasses)

    # Only attributes should have a default value
    assert not is_attr or "default_value" not in resolved_type_details

    if "Attr" in resolved_type_superclasses:
        resolved_type_superclasses.remove("Attr")

    base_type_name = "baseAttr" if is_attr else "baseType"
    base_type_str = resolved_type_details[base_type_name]["def"]
    base_attr, _ = resolve_value(json_in, arg_name, base_type_str, is_attr)

    if len(resolved_type_superclasses) != 1:
        print(f"{arg_name} has an unexpected number of superclasses.")
        sys.exit(1)

    single_attribute = resolved_type_superclasses[0]
    if single_attribute == f"Optional{'Attr' if is_attr else ''}":
        if base_attr.__class__ is not ValueType:
            print(f"{arg_name} Optional type is not supported by "
                  f"{base_attr.__class__}")
            sys.exit(1)
        base_attr.__class__ = OptionalValueType
        if is_attr:
            base_attr.updateDefaultValue('std::nullopt')
        return base_attr, is_attr

    if is_attr and single_attribute == "DefaultValuedAttr":
        base_attr.updateDefaultValue(resolved_type_details["defaultValue"])
        return base_attr, is_attr

    if not is_attr and single_attribute == "Variadic":
        if base_attr.__class__ is not ValueType:
            print(f"{arg_name} Variadic type is not supported by "
                  f"{base_attr.__class__}")
            sys.exit(1)
        base_attr.__class__ = VariadicValueType
        return base_attr, is_attr

    print(f"Attribute {single_attribute} not handled in {__file__}")
    sys.exit(1)


def vals_from_json(json_in, op_key, sub_key, allow_attributes=True):
    assert 'Poptorch_BasicOp' in json_in[op_key]["!superclasses"]

    arg_types = []

    for arg in json_in[op_key][sub_key]["args"]:
        arg_name = arg[1]
        arg_type_str = arg[0]["def"]

        arg_type, is_attr = resolve_value(json_in, arg_name, arg_type_str)
        if is_attr and not allow_attributes:
            raise ValueError(f"Attributes not permitted for {sub_key}")

        arg_types.append(arg_type)

    return arg_types


def checkDefaultValues(arg_types):
    is_defaultable = True

    for arg in reversed(arg_types):
        is_default = arg.default_value is not None

        if not is_defaultable and is_default:
            if isinstance(arg, OptionalValueType):
                arg.updateDefaultValue(None)
            else:
                return arg.name

        is_defaultable &= is_default

    return None


# Convert all the arguments from json to ValueType
def args_from_json(json_in, op_key):
    arg_types = vals_from_json(json_in, op_key, "arguments")

    failed_arg = checkDefaultValues(arg_types)
    if failed_arg is not None:
        print("Failed to default {failed_arg} in {op_key}")
        sys.exit(1)

    return arg_types


def returns_from_json(json_in, op_key):
    return vals_from_json(json_in, op_key, "results", False)


for key in json_in.keys():
    if key.startswith("Poptorch_"):
        if 'Poptorch_BasicOp' not in json_in[key]["!superclasses"]:
            continue

        op_name = json_in[key]["!name"].replace("Poptorch_", "")
        poptorch_ops[op_name] = {}

        # Convert the args into our little decl type system.
        poptorch_ops[op_name]["args"] = args_from_json(json_in, key)
        poptorch_ops[op_name]["returns"] = returns_from_json(json_in, key)

# Create the builder Cpp/Hpp file.
builder_call_translations = {
    "FLOAT_VEC": "const std::vector<float> &",
    "OPTIONAL_FLOAT_VEC": "std::optional<std::vector<float>>",
    "STRING_VEC": "const std::vector<const char *> &",
    "INT_VEC": "const std::vector<std::int32_t> &",
    "LONG_VEC": "const std::vector<std::int64_t> &",
    "OPTIONAL_LONG_VEC": "std::optional<std::vector<std::int64_t>>",
    "BOOL_VEC": "const std::vector<std::int64_t> &",  # Avoid packing issue
    "INT": "std::int32_t",
    "LONG": "std::int64_t",
    "OPTIONAL_LONG": "std::optional<std::int64_t>",
    "FLOAT": "float",
    "OPTIONAL_FLOAT": "std::optional<float>",
    "DOUBLE": "double",
    "OPTIONAL_DOUBLE": "std::optional<double>",
    "OPTIONAL_DOUBLE_VEC": "const std::optional<std::vector<double>>&",
    "STRING": "const char *",
    "OPTIONAL_STRING": "std::optional<const char *>",
    "TENSOR": "poptorch_ir::TensorId",
    "OPTIONAL_TENSOR": "poptorch_ir::OptionalTensorId",
    "TENSOR_VEC": "const std::vector<poptorch_ir::TensorId> &",
    "BOOL": "bool",
    "TYPE": "poptorch_ir::Type",
    "OPTIONAL_TYPE": "std::optional<poptorch_ir::Type>"
}

#"results": {"args": [
#    [{"def": "Poptorch_tensor", "kind": "def", "printable": "Poptorch_tensor"}, "result"],
#    [{"def": "Poptorch_tensor", "kind": "def", "printable": "Poptorch_tensor"}, "mean"],
#    [{"def": "Poptorch_tensor", "kind": "def", "printable": "Poptorch_tensor"}, "standard_deviation"]]

#define ARG(Name, Type) Type Name
#define BODY_ARG(Name) convert(Name, *_impl)

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
    func_args_decl = []
    parameters = []

    # Turn the args into function signatures.
    for arg in poptorch_ops[op_name]["args"]:
        new_arg = builder_call_translations[arg.macro_type()] + " " + arg.name
        func_args.append(new_arg)
        if arg.default_value is not None:
            new_arg += " = " + arg.default_value
        func_args_decl.append(new_arg)

        # If the argument is an MLIR type we want to convert it first.
        if arg.macro_type() == "TYPE":
            parameters.append("_impl->convertType(" + arg.name + ")")
        elif arg.macro_type() == "OPTIONAL_TYPE":
            parameters.append(
                arg.name +
                ".has_value() ? std::optional<mlir::Type>(_impl->convertType(*"
                + arg.name + ")) : std::nullopt")
        else:
            parameters.append("convert(" + arg.name + ", *_impl)")

    func_args_str = ", ".join(func_args)

    cppFunction = function_def.replace("__OP_NAME_PLACEHOLDER__",
                                       "PoptorchCompiler::" + op_name)
    headerFunction = function_def.replace("__OP_NAME_PLACEHOLDER__", op_name)

    cppFunction += func_args_str + ") {\n\n"

    # Create the IR op.
    cppFunction += "auto tmp = _impl->createOp<poptorch_ir::"
    cppFunction += op_name + ">("
    cppFunction += ", ".join(parameters) + ");\n\n"

    # Allow for each return to be optional, normal or variadic.
    cppFunction += "poptorch_ir::ODSTensorResults results;\n"

    for arg_num, arg in enumerate(poptorch_ops[op_name]["returns"]):
        cppFunction += "results.emplace_back();\n"
        cppFunction += "// Each result may be optional or variadic.\n"
        cppFunction += "for(mlir::Value value : tmp.getODSResults("
        cppFunction += str(arg_num) + ")) {\n"
        cppFunction += "  results.back().tensor_ids."
        cppFunction += "push_back(_impl->addValue(value));\n"
        cppFunction += "  results.back().requires_grad_types."
        cppFunction += "push_back(poptorch_ir::RequiresGradType::{});\n".format(
            arg.requires_grad_type())
        cppFunction += "}\n\n"

    cppFunction += "return results;\n"

    # Add the function end scope.
    cppFunction += "}\n\n"

    # Save the header file and implementation
    print(headerFunction + ", ".join(func_args_decl) + ");",
          file=parse_args.interface_header_file)
    print(cppFunction, file=parse_args.interface_cpp_file)
