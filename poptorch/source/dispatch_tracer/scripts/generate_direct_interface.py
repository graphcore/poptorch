# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# PyTorch Schema types to C++ convertor.
schemaToCpp = {
    "int[]": "toIntVector",
    "int[1]": "toIntVector",
    "int[2]": "toIntVector",
    "int[3]": "toIntVector",
    "int": "toInt",
    "int?": "toOptionalInt",
    "bool": "toBool",
    "float": "toDouble",
    # We treat all scalars as double for now.
    "Scalar": "toDouble",
    "Scalar?": "toOptionalDouble"
}


# Convert a tensor into it's constituent pieces.
def canonicalise_tensor(tensor):
    # Given a tensor type from the yml like Tensor(a!) we turn that into a nicer list format.
    # The format is [id, inplace, view]
    # Tensor(a!) -> [a, True, False]
    # Tensor(a) -> [a, False, True]
    # Tensor -> ["", False, False]
    # If a tensor is not inplace or a view there is nothing to refer to in the context of arguments and returns.
    # See the usage of these rules in native_functions.yml and in https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native for a full expansion of the precise rules (of which the above is an approximation).

    # We have a normal non-aliasing tensor.
    if '(' not in tensor:
        return ['', False, False]

    is_inplace = '!' in tensor

    # If there are brackets but no `!` then this is view of a tensor.
    is_view = not is_inplace

    # The id of the tensor. This is the identifier given to map an input onto an output.
    tensor_id = tensor[len('Tensor('):-1]

    # Remove the `!` if this is inplace.
    if is_inplace:
        assert tensor_id[-1] == '!'
        tensor_id = tensor_id[:-1]

    return [tensor_id, is_inplace, is_view]


def add_outplace_op(function, parameters, outputs, named_tensors, scope=""):
    # If it is mutliple outputs
    is_multiple_outputs = len(outputs) > 1

    # Return either a vector or single tensor.
    if is_multiple_outputs:
        return_type = "\tstd::vector<poptorch_ir::TensorId> mlir_output ="
    else:
        return_type = "\tpoptorch_ir::TensorId mlir_output ="

    # Generate the call to the compiler function.
    function_decl = "{}\t{} _compiler.".format(scope, return_type)
    function_decl += function + "(" + parameters + ");\n"

    # Clear the stack and add the outputs.
    function_decl += scope + "\t// Pop pytorch inputs from stack\n"
    function_decl += scope + "\tstack.clear();\n"

    # Add each of the outputs
    function_decl += scope + "\t// Push the new outputs onto the stack\n"

    index = 0

    # Handle each of the outputs.
    for output in outputs:
        # Capture all metadata related to each of the output tensors.
        output_info = canonicalise_tensor(output)
        tensor_id = output_info[0]
        is_inplace = output_info[1]
        is_view = output_info[2]

        # We either are expecting one tensor or a vector of tensors.
        if not is_multiple_outputs:
            output_id = "mlir_output"
        else:
            output_id = "mlir_output[" + str(index) + "]"
            index += 1

        # For each output tensor return it to pytorch in a different way depending on what the schema tells us.
        if is_inplace:
            # Inplace operations should be inplaced versions of a certain input.
            function_decl += "\tstack.push_back(outputIsInplaceOf("
            function_decl += output_id + ", " + named_tensors[tensor_id]
            function_decl += "_pytorch));\n"
        elif is_view:
            # Views should be a view of a given input.
            function_decl += "\tstack.push_back(outputIsViewOf(" + output_id
            function_decl += ", " + named_tensors[tensor_id]
            function_decl += "_pytorch, requires_grad));\n"
        else:
            # Otherwise we are returning a new tensor.
            function_decl += "\tstack.push_back(makeEmptyOutputTensor("
            function_decl += output_id + ", requires_grad));\n"

    return function_decl


def add_inplace_op(function, parameters, inplace_out, scope=""):
    # Generate the call to the compiler function.
    function_decl = scope + "\t_compiler." + function

    # Add the parameters.
    function_decl += "(" + parameters + ");\n"

    # Clear the stack and add the outputs.
    function_decl += scope + "\t// Pop pytorch inputs from stack and add the"
    function_decl += "inplaced tensor as the output.\n"

    # Add the inplace output to the stack.
    function_decl += "\t\tstack.clear();\n"
    function_decl += "\t\tstack.push_back(" + inplace_out + ");\n"
    return function_decl


# Some operations are maybe inplace or outplace depending on dynamic arguments. E.G
# add.out(in1, in2, out!) -> (out!)
# If in1 == out! it is inplace
# Otherwise it is outplace with a copy into out.
# The compiler will later optimize out the copy if it is not needed.
def add_maybe_inplace_op(op_target, parameters, tensor_params,
                         inplace_overload, named_tensors, unused_inplace_arg,
                         outputs):
    function_decl = ""
    for arg in unused_inplace_arg:
        # Add a comment to the autogenerated code to help explain it.
        function_decl += "\n\t// We need to check if a tensor is inplace on "
        function_decl += "one of its inputs. In that case we should be calling"
        function_decl += " an inplace operation directly.\n"

        # Check if each input is the inplace output.
        function_decl += "\tconst bool is_inplace_on_input = isInplaceOnInput("
        function_decl += arg + "_pytorch, {" + tensor_params + "});\n"
        function_decl += "\tif (is_inplace_on_input) {\n"

        # If one of the outputs is expecting to be inplace...
        function_decl += add_inplace_op(inplace_overload, parameters,
                                        arg + "_pytorch", "\t")
        function_decl += "\t} else {\n"

        # Otherwise we perfom the operation outplace.

        # Short comment.
        function_decl += "\t\t// Otherwise we should call the outplace variant"
        function_decl += "and manually mark the function.\n"

        # Add the out place version of the operation.
        function_decl += add_outplace_op(op_target["PopTorchDirect"],
                                         parameters, outputs, named_tensors,
                                         "\t")
        function_decl += "}\n"
    return function_decl


# Generate the c++ function which handles this operation.
def generate_cpp(op_target, canonicalised_args, outputs, named_tensors):
    # Some arguments we just completely ignore.
    args_to_ignore = {} if "IgnoreArgs" not in op_target else op_target[
        "IgnoreArgs"]

    # Some arguments are only marking the output tensor so we don't pass
    # them into the mlir call.
    key = "UnusedOutputArguments"
    unused_inplace_arg = {} if key not in op_target else op_target[key]

    # Some operations are an inplace operation. Their behaviour is slightly
    # more complex than just inplace. Sometimes it is inplace on one of it
    # inputs, sometimes it is inplace on another tensor which we treat as an
    # inplace + copy.
    key = "PopTorchDirectInplace"
    inplace_overload = None if key not in op_target else op_target[key]

    # We need to check whether or not a node actually requires a grad.
    function_decl = "\tbool requires_grad = false;\n"

    arg_index = 0
    parameters = " "
    tensor_params = ""
    inplace_ins = []

    for arg in canonicalised_args:
        # We just skip some arguments
        if arg[0] in args_to_ignore:
            arg_index += 1
            continue

        stack_at_index = "stack.at(" + str(arg_index) + ")"
        arg_type = arg[1]

        # Handle tensor list e.g "_cat(Tensor[] tensors, int dim=0) -> Tensor"
        if 'Tensor[]' in arg_type:
            # Error if this is not a list.
            function_decl += "ERROR_ON_MSG(!" + stack_at_index
            function_decl += ".isTensorList(), \"Expected list of tensors for"
            function_decl += " operation: {}\");\n".format(op_target['func'])

            # Create the list converted into poptorch compiler tensors.
            function_decl += "\t[[maybe_unused]] std::vector<"
            function_decl += "poptorch_ir::TensorId> " + arg[0] + ";\n"

            # Placeholder value to store each IValue in the list
            loop_placeholder = arg[0] + "_pytorch"

            # Iterate over the list.
            function_decl += "for (c10::IValue " + loop_placeholder + " : "
            function_decl += stack_at_index + ".toTensorVector()) {\n"

            # Extract the tensor from the list and look it up.
            function_decl += "\t\t" + arg[0] + ".push_back(findTensor("
            function_decl += "{}.toTensor()) );\n".format(loop_placeholder)

            # Update the requires grad stuff.
            function_decl += "\t\trequires_grad |= " + loop_placeholder
            function_decl += ".toTensor().requires_grad();\n\n"

            # end loop
            function_decl += "}"

        elif 'Tensor' in arg_type:
            # checking if Tensor exists or is none type
            if 'Tensor?' in arg_type:
                function_decl += "\t// Check if the optional tensor is none\n"
                function_decl += "\t// type, if false then access the tensor "
                function_decl += "passed = for operand\n\t bool " + arg[0]
                function_decl += "_pytorch_check = " + stack_at_index
                function_decl += ".isNone(); \n \tat::Tensor " + arg[0]
                function_decl += "_pytorch;\n\tif (!" + arg[
                    0] + "_pytorch_check)"
                function_decl += "{ " + arg[0] + "_pytorch = " + stack_at_index
                function_decl += ".toTensor();} \n"
            else:
                function_decl += "\t// Get the pytorch tensor, find the MLIR IR"
                function_decl += " mapped tensor for that tensor, and check if "
                function_decl += "this tensor needs a grad (if so, so does the "
                function_decl += "output).\n\tat::Tensor " + arg[0]
                function_decl += "_pytorch=" + stack_at_index + ".toTensor();\n"

            function_decl += "\t[[maybe_unused]] poptorch_ir::TensorId " + arg[
                0] + " = findTensor(" + arg[0] + "_pytorch);\n"
            function_decl += "\trequires_grad |= " + arg[
                0] + "_pytorch.requires_grad();\n\n"

            # All args should be [Name, type] but tensors with additional info will be [Name, Type, Id, inplace, view]
            if len(arg) > 2 and arg[3]:
                inplace_ins.append(arg[0])
        else:
            function_decl += "\tauto " + arg[0] + " = " + schemaToCpp[
                arg_type] + "(" + stack_at_index + ");\n"

        if arg[0] not in unused_inplace_arg:
            parameters += arg[0] + ","

            if 'Tensor' in arg_type:
                tensor_params += ", " + arg[0] + "_pytorch"
        arg_index += 1

    # Remove the comma
    parameters = parameters[:-1]

    if len(tensor_params) > 0:
        tensor_params = tensor_params[1:]

    # Check if the inplace operation is directly on one of the input tensors.
    # Some special case inplace operations have the form Op(in, in2, out!) with `out!` being the operation. This can mean that either out! is an alias of `in` or that it is a completely seperate tensor. For now
    if len(unused_inplace_arg) == 1 and inplace_overload:
        function_decl += add_maybe_inplace_op(op_target, parameters,
                                              tensor_params, inplace_overload,
                                              named_tensors,
                                              unused_inplace_arg, outputs)
    elif "PopTorchDirect" in op_target:
        # Otherwise we are dealing with a vanilla function.
        function_decl += add_outplace_op(op_target["PopTorchDirect"],
                                         parameters, outputs, named_tensors)
    else:
        function_decl += add_inplace_op(op_target["PopTorchDirectInplace"],
                                        parameters,
                                        inplace_ins[0] + "_pytorch")

    function_decl += "}\n"
    return function_decl


class DirectMlirGenerator:
    def __init__(self, header_file, cpp_file, lookup_file):

        # The files to output the results into.
        self.header = header_file
        self.cpp = cpp_file
        self.lookup = lookup_file

    def gen_function(self, function_name, op_target, arguments, outputs):
        # We convert the args from the schema string to a list of lists.
        # The outer list being all arguments and the inner being the information
        # for that given argument. Most types are just [Name, Type] pairs in the list
        # however tensors can be views or inplace so we track that.
        canonicalised_args = []

        # Tensors which have been marked as being inplace/views will have an ID.
        named_tensors = {}
        # Remove any optional value assignment (`=`) and take the last string which will be the name.
        for argument in arguments:
            argument = argument.strip()

            # The argument '*' is a special case for the python interface, we can ignore.
            if argument == "*":
                continue

            # Here we are processing the arguments from native functions.yml, i.e:
            # aten.contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)

            # Remove default arguments.
            arg_name = argument.split('=')[0].split(' ')[-1]

            # E.g Tensor(a) self -> name: self, type : Tensor(a)
            arg_name = arg_name.split(' ')[-1]
            arg_type = argument.split(' ')[0]

            # Add the special tensor information. Some tensors don't have any, if there is a
            # `(` it means there is view or inplace information.
            # Tensor -> just a tensor.
            # Tensor(a) -> view of `a`
            # Tensor(a!) -> `a` modified inplace
            if 'Tensor' in arg_type and '(' in arg_type:
                # Turn the tensor into [Name, 'Tensor', id, is_inplace, is_view]
                canonical = [arg_name, 'Tensor'
                             ] + canonicalise_tensor(arg_type)

                named_tensors[canonical[2]] = arg_name
                canonicalised_args.append(canonical)
            else:
                # Just add a normal list.
                canonicalised_args.append([arg_name, arg_type])

        aten_name = function_name
        function_name = function_name.replace('.', '_')

        # Generate the C++ impl.
        function_decl = "void MLIRDispatch::" + function_name
        function_decl += "(c10::Stack& stack) {\n"
        function_decl += generate_cpp(op_target, canonicalised_args, outputs,
                                      named_tensors)

        # Print the C++ impl.
        print(function_decl, file=self.cpp)

        # Generate the C++ header.
        print("void " + function_name + "(c10::Stack& stack);\n",
              file=self.header)

        # Generate the Aten Op to the C++ function map.
        print("{\"aten::" + aten_name + "\", [=](c10::Stack& stack) { this->" +
              function_name + "(stack);}},\n",
              file=self.lookup)
