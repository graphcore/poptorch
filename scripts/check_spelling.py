#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import ast
import glob
import os
import re
import shlex
import shutil
import signal
import sys
import termios
import threading
import tty

from utils import _utils

CUSTOM_DIC_PATH = "docs/common/custom_dic"

HUNSPELL_CMD = [
    "hunspell",
    "-a",  # Pipe mode
    "-d",
    "en_GB",  # Graphcore uses en_GB for documentation
    "-i",
    "utf-8",  # Encoding: suitable for linux and osx
    "-mode=none"
]  # Use raw text

TERM_STDIN = sys.stdin


def getChar():
    try:
        # Backup this or the terminal will break on closing
        old_attr = termios.tcgetattr(TERM_STDIN.fileno())
        tty.setraw(TERM_STDIN.fileno())
        char = TERM_STDIN.read(1)
    finally:
        # Reset the terminal
        termios.tcsetattr(TERM_STDIN.fileno(), termios.TCIFLUSH, old_attr)
    return char


class DocStr():
    def __init__(self, doc_str, source_file, line_num):
        self._doc_str = doc_str
        self._source_file = source_file
        self._line_num = line_num

    @property
    def doc_str(self):
        return self._doc_str

    @property
    def line_num(self):
        return self._line_num

    @property
    def source_file(self):
        return self._source_file

    def __str__(self):
        s = f"{self._line_num}:" + self._doc_str
        return s


def start_hunspell_process():
    # Add custom dictionary first time only
    if "-p" not in HUNSPELL_CMD:
        custom_dic_path = os.path.join(_utils.sources_dir(), CUSTOM_DIC_PATH)

        if not os.path.exists(custom_dic_path):
            open(custom_dic_path, 'a').close()

        HUNSPELL_CMD.append("-p")
        HUNSPELL_CMD.append(shlex.quote(custom_dic_path))

    hunspell_output = []

    def out_handler(line):
        hunspell_output.append(line)

    # subprocess.Popen fails to pass the filename correctly without this when
    # shell=True. shlex.quote will handle any spaces correctly.
    cmd = " ".join(HUNSPELL_CMD)

    hunspell_proc = _utils.Process(cmd,
                                   env=None,
                                   redirect_stderr=True,
                                   stdout_handler=out_handler,
                                   bufsize=0)

    # First line is just a version
    while len(hunspell_output) < 1:
        assert hunspell_proc.is_running()
    hunspell_output.clear()

    return {'proc': hunspell_proc, 'out': hunspell_output}


CODE_BLOCK = re.compile(r"\.\. code-block::[^\n]+\n\n.*?\n\n", flags=re.DOTALL)


def strip_code_blocks(s):
    s_list = list(s)
    for match in CODE_BLOCK.finditer(s):
        for pos in range(match.start(), match.end()):
            # Preserve lines by replacing everything except new lines with
            # spaces
            if s_list[pos] != "\n":
                s_list[pos] = " "
    return "".join(s_list)


def should_skip(line):
    stripped_line = line.strip()
    if stripped_line.startswith(">>>"):
        return True

    if stripped_line.startswith("..."):
        return True

    return False


ALL_EXCLUSIONS = (re.compile(r":param [^:]+:"), re.compile(r"p[0-9]+[^0-9]"),
                  re.compile(r":py:[^:]+:"), re.compile(r"T[0-9]+[^0-9]"),
                  re.compile(r"`+[^`]+`+"), re.compile(r":r?type.*"))


def remove_exclusions(line):
    for exclusion in ALL_EXCLUSIONS:
        line = exclusion.sub("", line)

    line = line.replace(".. seealso::", "")

    return line


def get_doc_str_line_number(element):
    # Handle the case of lots of parameters etc
    if isinstance(element.body[0], ast.Expr):
        if isinstance(element.body[0].value, ast.Str):
            end_line_no = element.body[0].value.lineno
            doc_str_lines = element.body[0].value.s.count("\n")
            return end_line_no - doc_str_lines

    # If the string lookup fails
    return element.lineno


DOC_STR_ELEMENTS = (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef,
                    ast.Module)


def recursive_add_doc_str(source_file, element, doc_str_list):

    for sub_element in element.body:
        if isinstance(sub_element, DOC_STR_ELEMENTS):
            doc_str = ast.get_docstring(sub_element)

            if doc_str is not None:
                doc_str_list.append(
                    DocStr(doc_str, source_file,
                           get_doc_str_line_number(sub_element)))

        if hasattr(sub_element, "body"):
            recursive_add_doc_str(source_file, sub_element, doc_str_list)


BLACK_ON_WHITE = "\033[30;107m"
RESET_COLOR = "\033[39;49m"
UNDERLINE = "\033[4m"
NOT_UNDERLINE = "\033[24m"


def print_context(doc_str, line_offset, unknown_spelling):
    print(BLACK_ON_WHITE, end='')

    all_lines = doc_str.doc_str.split("\n")

    for line_num, line in enumerate(all_lines):
        if line_num == line_offset:

            # Make sure we find the right incident of spelling
            pattern = unknown_spelling + r"[^a-z]"
            match_start = re.search(pattern, line + " ").start()

            before = line[:match_start]
            print(before, end='')

            print(UNDERLINE, end='')
            print(unknown_spelling, end='')
            print(NOT_UNDERLINE, end='')

            after = line[match_start + len(unknown_spelling):]
            print(after, end='')
        else:
            print(line, end='')

        if line_num + 1 != len(all_lines):
            print()

    print(RESET_COLOR + "\n")


def process_incorrect_word(hunspell, result, doc_str, line_offset):
    result = result.split(" ")

    symbol = result[0]
    if symbol not in ("&", "#"):
        raise RuntimeError("Invalid symbol")

    unknown_spelling = result[1]

    line_num = doc_str.line_num + line_offset

    while True:
        print_context(doc_str, line_offset, unknown_spelling)
        print(f"Unknown spelling, '{unknown_spelling}' on line {line_num}" +
              f" ({doc_str.source_file}).")

        if symbol == b"&":
            # Comma seprated list of suggestions
            suggestions = [r.decode("utf-8") for r in result[4:]]
            print("Suggestions: " + " ".join(suggestions))

        print("(space): continue, (a)dd to dictionary, (q)uit")
        c = getChar()

        if c == ' ':
            break
        if c == 'a':
            # Add to dictionary and save
            hunspell['proc'].write(b"*")
            hunspell['proc'].write(unknown_spelling.encode("utf-8"))
            hunspell['proc'].write(b"\n")
            hunspell['proc'].write(b"#\n")
            break
        # Ctrl+c and ctrl+z are intercepted
        if c in ('q', '\x03', '\x04'):  # ^C and ^D
            sys.exit(0)
        if c == '\x1a':  # ^Z
            signal.pthread_kill(threading.get_ident(), signal.SIGSTOP)

    print("\n\n\n\n")


def process_doc_str(hunspell, doc_str):
    all_doc_str = doc_str.doc_str
    all_doc_str = strip_code_blocks(all_doc_str)

    all_lines = all_doc_str.split("\n")
    for line_offset, line in enumerate(all_lines):
        if should_skip(line):
            continue

        line = remove_exclusions(line)

        full_line = b"^"  # Escape any commands
        full_line += line.encode('utf-8') + b"\n"

        hunspell['proc'].write(full_line)

        while True:
            if len(hunspell['out']) == 0:
                assert hunspell['proc'].is_running()
                continue

            next_token = hunspell['out'].pop(0)
            if next_token == "":
                break

            if (next_token == "*" or next_token == "-"
                    or next_token[0] == "+"):
                continue
            process_incorrect_word(hunspell, next_token, doc_str, line_offset)


def check_source_file(source_dir, source_file):
    source_file_without_root = source_file[len(source_dir) + 1:]
    print(f"Checking {source_file_without_root}\n")

    with open(source_file, 'r') as f:
        source = f.read()

    ast_module = ast.parse(source, source_file)

    all_doc_str = []
    recursive_add_doc_str(source_file_without_root, ast_module, all_doc_str)

    hunspell = start_hunspell_process()

    for doc_str in all_doc_str:
        process_doc_str(hunspell, doc_str)

    hunspell['proc'].eof()
    hunspell['proc'].wait()


if __name__ == "__main__":
    if _utils.get_os_type() != _utils.OsType.Linux:
        print("Not running on linux.")
        sys.exit(1)

    if shutil.which(HUNSPELL_CMD[0]) is None:
        print(f"Please install {HUNSPELL_CMD[0]}.")
        sys.exit(1)

    source_dir = os.path.join(_utils.sources_dir(), "python")

    for source_file in glob.glob(os.path.join(source_dir, "*.py")):
        check_source_file(source_dir, source_file)
