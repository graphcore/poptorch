#! /usr/bin/python

import clang.cindex
import sys
import json

jsonOutput = {}

files = [sys.argv[1] + "builder.hpp", sys.argv[1] + "builder.h.gen"]

nodeBlacklist = {"DomainOpSet", "Builder", "getOpsetVersion", "AiOnnxOpset10", "AiOnnxOpset11"}

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
            find_functions(c,namespace)

    if len(operation) > 0:

        if namespace not in jsonOutput:
            jsonOutput[namespace] = {}
        jsonOutput[namespace][node.spelling] = operation


index = clang.cindex.Index.create()


tu = index.parse(sys.argv[1] + "builder.hpp", args=["-std=c++14",
                                            "-I/Users/stephenm/Projects/popart_install/include/",
                                            "-I/usr/local/Cellar/llvm/8.0.0_1/include/c++/v1/",
                                            "-I/Library/Developer/CommandLineTools/usr/lib/clang/10.0.1/include/"])

for diag in tu.diagnostics:
    print(diag)
root_node = tu.cursor
find_functions(root_node,"")

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



# Convert the raw C++ type parsed from the header into the macro type.
def toType(cxxType):

    if "boost::optional" in cxxType:
        return "UNKNOWN"

    if cxxType == "int64_t":
        return "INT"

    if "int64_t" in cxxType:
        return "INT_VEC"

    if cxxType == "unsigned int":
        return "INT"


    if cxxType == "float":
        return "FLOAT"

    return "UNKNOWN"


def attrTypeGetter(cxxType):
    if cxxType == "int64_t":
        return "i"

    if "int64_t" in cxxType:
        return "is"

    if cxxType == "unsigned int":
        return "i"

    if cxxType == "float":
        return "f"




macroFile = ""

headerStubs = ""

cxxFile = ""

for opset in classes:
    for name in jsonOutput[opset]:
        # Generate the macro

        opDecl = "OP_DECL("

        opDecl += "\"popart::" + name + "\", " + name

        if opset.startswith("AiOnnxOpset"):
            opDecl += ", AiOnnxOpset9." + name
        else:
            opDecl += ", " + opset + "." + name


        argVector = ""
        bodyArgVector = ""

        earlyExit = False
        args = jsonOutput[opset][name]["args"]
        for arg in args:
            # Skip the first args and also the "name" arg.
            if arg["name"] == "args" or arg["name"] == "name":
                continue

            macroType = toType(arg["type"])

            if macroType == "UNKNOWN":
                earlyExit = True
                break

            argVector += "ARG(" + macroType + "," + arg["name"] + ") "

            bodyArgVector += "BODY_ARG(" + arg["name"] + ") "

        if earlyExit:
            continue


        if argVector == "":
            argVector = "NONE"

        if bodyArgVector == "":
            bodyArgVector = "NONE"

        opDecl += ", " + argVector
        opDecl += ", " + bodyArgVector

        # Return type handler.
        if "std::vector" in jsonOutput[opset][name]["type"]:
            opDecl += ", [0])"
        else:
            opDecl += ", NONE)"

        macroFile += opDecl + "\n"


        header = "torch::jit::Node* "

        header += "Create_" + name + "(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args"


        cppFile = " torch::jit::Node *newNode = graph.create(c10::Symbol::fromQualString(\"popart::" + name + "\"), args);\n"

        args = jsonOutput[opset][name]["args"]
        for arg in args:
            # Skip the first args and also the "name" arg.
            if arg["name"] == "args" or arg["name"] == "name":
                continue

            header += "," + arg["type"] + " " + arg["name"]

            attr = attrTypeGetter(arg["type"])

            cppFile += "newNode->" + attr + "_(c10::Symbol::fromQualString(\"attr::" +arg["name"]  + "\")," + arg["name"] + ");\n"


        cppFile += "return newNode;\n"

        cppFile = header + ") {\n" + cppFile + "}"

        header += ");"


        headerStubs += header + "\n"

        cxxFile += cppFile + "\n"




autogeneratedComment = "// Auto generated file, do not modify\n// Run `python3 PopParse.py to regenerate\n"
with open('CompilerOperationMacros.inc', 'w') as f:
    print(autogeneratedComment, file=f)
    print(macroFile, file=f)

with open('CompilerOps.h.inc', 'w') as f:
    print(autogeneratedComment, file=f)
    print(headerStubs, file=f)


with open('CompilerOps.cpp.inc', 'w') as f:
    print(autogeneratedComment, file=f)
    print(cxxFile, file=f)
