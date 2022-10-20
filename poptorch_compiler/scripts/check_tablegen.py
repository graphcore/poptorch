# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
Check that the PyTorch op definitions in native_functions.yaml match the
TableGen definitions.
"""

import copy
import itertools
import json
import re
import string
import sys
import yaml

TYPE_MAPPINGS = {
    'Tensor': [
        'Poptorch_tensor', 'Poptorch_tensor_no_grad', 'Poptorch_float_tensor',
        'Poptorch_non_boolean_tensor', 'Poptorch_integral_tensor'
    ],
    'Tensor[]': ['Poptorch_tensorlist'],
    'int': ['I64Attr'],
    'int[]': ['I64ArrayAttr'],
    'bool': ['BoolAttr'],
    'bool[]': ['I64ArrayAttr'],
    'Scalar': ['F32Attr', 'F64Attr'],
    'float': ['F32Attr', 'F64Attr'],
    'float[]': ['F32ArrayAttr'],
    'str': ['StrAttr'],
    'ScalarType': ['TypeAttr'],
}

native_functions = sys.argv[1]
td_json = sys.argv[2]
yml_files = sys.argv[3:]

# Parse the native functions
with open(native_functions, "r") as f:
    y_native_functions = yaml.safe_load(f)
function_prototypes = [x['func'] for x in y_native_functions]
func_re = re.compile(r"([^\(]*)\((.*)\) -> (.*)")
index_re = re.compile(r"\[[0-9]+\]")
bracketedname_bang_re = re.compile(r"\([^\)]*!\)")
bracketedname_re = re.compile(r"\([^\)]*\)")


def has_more_close_braces(str, open='[', close=']'):
    count = 0
    for c in str:
        if c == open:
            count += 1
        elif c == close:
            count -= 1
    return count < 0


def process_types(types, extract_names=False):
    if len(types) == 1 and types[0] == '':
        return []
    types = [x.strip() for x in types]
    types = [x for x in types if not x == '*' and not has_more_close_braces(x)]
    has_default = ["=" in x.rpartition(" ")[2] for x in types]
    types = [x.rsplit(" ", 1) for x in types]
    if extract_names:
        names = [x[1].strip().rsplit('=', 1)[0] for x in types]
    types = [x[0] for x in types]
    types = [index_re.sub("[]", x) for x in types]
    types = [bracketedname_bang_re.sub("?", x) for x in types]
    types = [bracketedname_re.sub("", x) for x in types]
    types = [x + "?" if y else x for x, y in zip(types, has_default)]
    if extract_names:
        types = list(zip(types, names))
    return types


functions = {}
for f in function_prototypes:
    m = func_re.fullmatch(f)
    if m is None:
        raise RuntimeError("Couldn't parse function prototype {}".format(f))
    fname = m.group(1).strip()
    params = process_types(m.group(2).split(","), True)
    ret = m.group(3).strip(string.whitespace)
    if ret.startswith("(") and ret.endswith(")"):
        ret = ret[1:len(ret) - 1]
    return_val = process_types(ret.split(","))

    functions[fname] = {'params': params, 'return': return_val}

# Parse the TableGen json
tblgen = {}
with open(td_json, "r") as f:
    j_td = json.load(f)
    for f in (x for x in j_td if x.startswith("Poptorch_")):
        fname = f[len("Poptorch_"):]
        if not all(x in j_td[f] for x in ["arguments", "results"]):
            continue
        params = [x[0]['def'] for x in j_td[f]["arguments"]["args"]]
        results = [x[0]['def'] for x in j_td[f]["results"]["args"]]

        def resolveType(t):
            if t not in j_td:
                return t
            if t.startswith("Poptorch"):
                return t
            attr = 'baseAttr' if 'baseAttr' in j_td[t] else 'baseType'
            if attr not in j_td[t]:
                return t
            if j_td[t][attr] is None:
                return t
            resolved = j_td[t][attr]['def']
            if "OptionalAttr" in j_td[t]["!superclasses"]:
                return resolved + "?"
            return resolved

        params = [resolveType(x) for x in params]
        results = [resolveType(x) for x in results]

        tblgen[fname] = {'params': params, 'results': results}

# Parse the PopTorch YAML files and check they match
failed = False
for yf in yml_files:
    with open(yf, "r") as f:
        y_yml_file = yaml.safe_load(f)
    for entry in y_yml_file:
        # Extract the information and look up the tblgen definition
        fname = entry['func'].strip()
        if fname not in functions:
            raise RuntimeError(
                "Function {} from {} does not exist in native_functions.yml".
                format(fname, yf))
        if "PopTorchDirect" not in entry:
            raise RuntimeError(
                "Didn't find PopTorchDirect in function entry {} in {}".format(
                    fname, yf))
        if "tblgen" in functions[fname]:
            raise RuntimeError("Duplicate function entry {}".format(fname))
        tblgen_func = entry["PopTorchDirect"]
        functions[fname]["tblgen"] = tblgen_func
        if tblgen_func not in tblgen:
            raise RuntimeError(
                "TableGen function {} not found".format(tblgen_func))
        pytorch_params = functions[fname]['params']
        pytorch_results = functions[fname]['return']
        tblgen_params = tblgen[tblgen_func]['params']
        tblgen_results = tblgen[tblgen_func]['results']

        def to_set(dict_or_set):
            if isinstance(dict_or_set, dict):
                return set(dict_or_set.keys())
            if isinstance(dict_or_set, set):
                return dict_or_set
            return {dict_or_set}

        ignored_args = set()
        if 'IgnoredArgs' in entry:
            ignored_args.update(to_set(entry['IgnoredArgs']))
        if 'UnusedOutputArguments' in entry:
            ignored_args.update(to_set(entry['UnusedOutputArguments']))

        def filter_ignored_pytorch_args(pt, ignored_args):
            pt_copy = []
            ignored_args_copy = copy.copy(ignored_args)
            for ptt_type, ptt_name in pt:
                if ptt_name in ignored_args_copy:
                    ignored_args_copy.remove(ptt_name)
                else:
                    pt_copy.append(ptt_type)
            return pt_copy, ignored_args_copy

        filtered_pytorch_params, unremoved_args = filter_ignored_pytorch_args(
            pytorch_params, ignored_args)
        if len(unremoved_args) > 0:
            failed = True
            print("Failed function:")
            print("")
            print(fname)
            print("  PyTorch params: {}".format(pytorch_params))
            print(f"  Ignored Args: {ignored_args}")
            print(f"  Missing Ignored Args: {unremoved_args}")
            print("")
            continue

        def check_types(pt, tg):
            mismatches = []
            for ptt, tgt in itertools.zip_longest(pt, tg):
                tgtOptional = False
                if tgt is not None and tgt.endswith("?"):
                    tgt = tgt[:len(tgt) - 1]
                    tgtOptional = True
                if ptt is None:
                    if not tgtOptional:
                        mismatches.append((None, tgt))
                    continue
                pttOptional = False
                if '?' in ptt:
                    pttOptional = True
                    ptt = ptt.replace('?', '')
                if pttOptional and tgt is None:
                    break
                mapped_types = TYPE_MAPPINGS.get(ptt)
                if mapped_types is None:
                    raise RuntimeError("No type mapping for {}".format(ptt))
                if tgt not in mapped_types:
                    mismatches.append((ptt, tgt))
            return mismatches

        mismatches = []
        try:
            mismatches.extend(
                check_types(filtered_pytorch_params, tblgen_params))
            mismatches.extend(check_types(pytorch_results, tblgen_results))
        except RuntimeError:
            print("Failed function:")
            print("")
            print(fname)
            print("  PyTorch params: {}".format(filtered_pytorch_params))
            print("  TableGen params: {}".format(tblgen_params))
            print("  PyTorch results: {}".format(pytorch_results))
            print("  TableGen results: {}".format(tblgen_results))
            print("")
            raise

        if len(mismatches) > 0:
            failed = True
            print(fname)
            print("  PyTorch params: {}".format(pytorch_params))
            print("  TableGen params: {}".format(tblgen_params))
            print("  PyTorch results: {}".format(pytorch_results))
            print("  TableGen results: {}".format(tblgen_results))
            print("  Mismatches: {}".format(mismatches))
            print("")

if failed:
    sys.exit(1)
