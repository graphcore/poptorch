#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import argparse
import clang.cindex
from ctypes.util import find_library
import json
import logging
import os
import sys

logger = logging.getLogger("PopParse")
parser = argparse.ArgumentParser()
parser.add_argument("-c",
                    "--clang",
                    type=str,
                    help="Manually set path to clang headers")
parser.add_argument("-D",
                    "--debug",
                    action='store_true',
                    help="Enable debug printing")

args = parser.parse_args()

logging_level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(level=logging_level)

if args.clang:
    clang.cindex.Config.set_library_file(args.clang)
else:
    clang.cindex.Config.set_library_file(find_library('clang-8'))

jsonOutput = {}

current_dir = os.path.dirname(os.path.realpath(__file__))

# Popart uses both relative includes and full "popart/XXX.hpp" includes so we need point clang to both paths.
popart_include_dir_partial = current_dir + "/../popart/popart/willow/include/"
popart_dir = popart_include_dir_partial + "popart"

files = [popart_dir + "/builder.hpp", popart_dir + "/builder.h.gen"]

nodeBlacklist = {
    "DomainOpSet", "Builder", "getOpsetVersion", "AiOnnxOpset10",
    "AiOnnxOpset11"
}


def find_children(node, argNum):
    argDict = {}
    if node.kind == clang.cindex.CursorKind.PARM_DECL:
        argDict["type"] = node.type.spelling
        argDict["name"] = node.spelling

    return argDict


def find_functions(node, namespace):
    global jsonOutput
    # If this is not the file path provided on the comand line, skip.
    if node.location.file != None and str(node.location.file) not in files:
        return
    if node.spelling in nodeBlacklist:
        return

    if node.kind == clang.cindex.CursorKind.CLASS_DECL:
        namespace = node.spelling

    operation = {}
    if node.kind == clang.cindex.CursorKind.CXX_METHOD:
        returnType = str(node.type.spelling).split("(")[0]

        operation["type"] = returnType
        operation["args"] = []

        #line += "OP_DEF(\"" + namespace + "." + node.spelling + "\", " + returnType + ", " + namespace + "::" + node.spelling + "("
        argNum = 0
        for c in node.get_children():

            # Traverse the children to find any parameters
            arg = find_children(c, argNum)

            if len(arg) > 0:
                arg["num"] = argNum
                operation["args"].append(arg)
                argNum += 1

    else:
        for c in node.get_children():
            find_functions(c, namespace)

    if len(operation) > 0:

        if namespace not in jsonOutput:
            jsonOutput[namespace] = {}
        jsonOutput[namespace][node.spelling] = operation


index = clang.cindex.Index.create()

tu = index.parse(popart_dir + "/builder.hpp",
                 args=[
                     "-std=c++14",
                     "-I" + popart_dir,
                     "-I" + popart_include_dir_partial,
                     "-DONNX_NAMESPACE=onnx",
                     "-I/usr/local/Cellar/llvm/8.0.0_1/include/c++/v1/",
                 ])

for diag in tu.diagnostics:
    logger.warn(diag)

root_node = tu.cursor
find_functions(root_node, "")
logger.debug("jsonOutput Keys:%s" % jsonOutput.keys())

classes = []

for n in jsonOutput:
    if n.startswith("Ai"):
        classes.append(n)

classes.reverse()

addedFunctions = set()

for opset in classes:

    toRemove = []

    for name in jsonOutput[opset]:
        if name in addedFunctions:
            toRemove.append(name)
        else:
            addedFunctions.add(name)

    for name in toRemove:
        jsonOutput[opset].pop(name)
logger.debug("addedFunctions: %s" % addedFunctions)

MultipleOutputsOps = {"lstm": "2", "split": "num_outputs", "topk": "2"}

CXXTypeToTypeClass = {
    # Scalar integers
    "int64_t": "INT",
    "int": "INT",
    "bool": "INT",
    "unsigned int": "INT",
    "popart::ReductionType": "INT",
    "nonstd::optional<int64_t>": "INT",
    "nonstd::optional<int>": "INT",

    # Floats
    "float": "FLOAT",
    "nonstd::optional<float>": "FLOAT",

    # Non-scalar floats
    "std::vector<float>": "FLOAT_VEC",

    # Non-scalar integers.
    "std::vector<int64_t>": "INT_VEC",
    "nonstd::optional<std::vector<int64_t> >": "INT_VEC"
}


# Convert the raw C++ type parsed from the header into the macro type.
def toType(cxxType):

    cleaned = cxxType.replace("&", "").replace("const", "").strip().rstrip()

    if cleaned in CXXTypeToTypeClass:
        return CXXTypeToTypeClass[cleaned]

    logger.debug(
        "toType: Unknown cxxType=%s / cleaned=%s" % (cxxType, cleaned))

    # Soft fail as it isn't unexpected for some popart functions to be unsupported right now.
    return "UNKNOWN"


# Convert from the popart header types into normal C++ types that can be used by pytorch.
def convertCxxConvert(cxxType):

    if "nonstd::optional<int>" in cxxType or "nonstd::optional<int64_t>" in cxxType:
        return "std::int32_t"

    if "popart::ReductionType" in cxxType:
        return "std::int32_t"

    if "nonstd::optional<float>" in cxxType:
        return "float"

    if "nonstd::optional<std::vector<int64_t" in cxxType:
        return "std::vector<int64_t>"

    # Most types won't need processing
    return cxxType


def attrTypeGetter(ty):
    if ty == "INT":
        return "i"

    if ty == "INT_VEC":
        return "is"

    if ty == "FLOAT":
        return "f"

    assert False, "Invalid type: " + ty


macroFile = ""

headerStubs = ""

cxxFile = ""

for opset in classes:
    for name in jsonOutput[opset]:
        # Generate the macro
        opDecl = "OP_DECL("

        opDecl += "popart, " + name + ", " + name

        if opset.startswith("AiOnnxOpset"):
            opDecl += ", AiOnnxOpset9." + name
        else:
            opDecl += ", " + opset + "." + name

        argVector = ""
        bodyArgVector = ""

        earlyExit = True
        args = jsonOutput[opset][name]["args"]
        for arg in args:
            # Skip the first args and also the "name" arg.
            if arg["name"] == "args":
                # Guarantee we are working with an op which takes in popart tensors as 0th argument.
                earlyExit = False
                continue

            if arg["name"] == "name":
                continue

            macroType = toType(arg["type"])

            if macroType == "UNKNOWN":
                logger.debug("Skipping OP: " + name +
                             " due to parse failure on " + str(arg))
                earlyExit = True
                break

            argVector += "ARG(" + macroType + "," + arg["name"] + ") "

            if "ReductionType" in arg["type"]:
                bodyArgVector += "BODY_ARG(static_cast<popart::ReductionType>(" + arg[
                    "name"] + ")) "
            else:
                bodyArgVector += "BODY_ARG(" + arg["name"] + ") "

        if earlyExit:
            continue

        if argVector == "":
            argVector = "NONE"

        if bodyArgVector == "":
            bodyArgVector = "NONE"

        opDecl += ", " + argVector
        opDecl += ", " + bodyArgVector

        macroFile += opDecl + ")\n"

        header = "torch::jit::Node* "

        header += "Create_" + name + "(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args"

        cppFile = " torch::jit::Node *newNode = graph.create(Symbols::popart::" + name + ", args"
        if name in MultipleOutputsOps:
            cppFile += ", %s" % MultipleOutputsOps[name]
        cppFile += ");\n"

        args = jsonOutput[opset][name]["args"]
        for arg in args:
            # Skip the first args and also the "name" arg.
            if arg["name"] == "args" or arg["name"] == "name":
                continue

            header += "," + convertCxxConvert(arg["type"]) + " " + arg["name"]

            attr = attrTypeGetter(toType(arg["type"]))

            cppFile += "newNode->" + attr + "_(c10::Symbol::fromQualString(\"attr::" + arg[
                "name"] + "\")," + arg["name"] + ");\n"

        cppFile += "graph.insertNode(newNode);\n"
        cppFile += "return newNode;\n"

        cppFile = header + ") {\n" + cppFile + "}"

        header += ");"

        headerStubs += header + "\n"

        cxxFile += cppFile + "\n"

autogeneratedComment = "// Copyright (c) 2020 Graphcore Ltd. All rights reserved.\n// Auto generated file, do not modify\n// Run `python3 PopParse.py to regenerate\n// clang-format off\n"
with open(
        os.path.join(current_dir, 'popart_compiler', 'include',
                     'popart_compiler', 'CompilerOperationMacros.inc'),
        'w') as f:
    print(autogeneratedComment, file=f)
    print(macroFile, file=f)

with open(
        os.path.join(current_dir, 'poptorch', 'include', 'poptorch',
                     'CompilerOps.h.inc'), 'w') as f:
    print(autogeneratedComment, file=f)
    print(headerStubs, file=f)

with open(
        os.path.join(current_dir, 'poptorch', 'source', 'CompilerOps.cpp.inc'),
        'w') as f:
    print(autogeneratedComment, file=f)
    print(cxxFile, file=f)
