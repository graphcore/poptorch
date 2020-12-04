# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import datetime as dt
import logging
import os
import sys
import subprocess
import traceback
import poptorch.poptorch_core as poptorch_core

# Create a poptorch logger which outputs to the console INFO messages and above
logger = logging.getLogger("poptorch::python")

_LOG_LEVEL_MAPPING = {
    "TRACE": (0, logging.DEBUG),
    "TRACE_ALL": (0, logging.DEBUG),
    "DEBUG": (1, logging.DEBUG),
    "DEBUG_IR": (1, logging.DEBUG),
    "INFO": (2, logging.INFO),
    "WARN": (3, logging.WARN),
    "ERR": (4, logging.ERROR),
    "OFF": (6, logging.CRITICAL)
}


def setLogLevel(level):
    if isinstance(level, int):
        # Legacy usage
        for key in _LOG_LEVEL_MAPPING:
            if _LOG_LEVEL_MAPPING[key][0] == level:
                setLogLevel(key)
                return

        raise ValueError("Invalid log level integer")

    try:
        # Change it in C++ first
        level_int = _LOG_LEVEL_MAPPING[level][0]
        poptorch_core.setLogLevel(level_int)

        # Then in python
        level_py = _LOG_LEVEL_MAPPING[level][1]
        logger.setLevel(level_py)
    except KeyError:
        error_str = "Unknown log level: " + str(level) + ". Valid values are "

        all_keys = sorted(list(_LOG_LEVEL_MAPPING.keys()))

        for key in all_keys:
            error_str += key
            if key == all_keys[-2]:
                error_str += " and "
            elif key != all_keys[-1]:
                error_str += ", "

        raise ValueError(error_str)


setLogLevel(os.environ.get("POPTORCH_LOG_LEVEL", "WARN"))


class _PoptorchFormatter(logging.Formatter):
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)
    RESET_COLOR = "\033[0m"
    BOLD_COLOR_SEQ = "\033[1;%dm"
    COLOR_SEQ = "\033[%dm"
    FORMATS = {
        logging.DEBUG: COLOR_SEQ % CYAN,
        logging.INFO: RESET_COLOR,
        logging.WARNING: BOLD_COLOR_SEQ % YELLOW,
        logging.ERROR: BOLD_COLOR_SEQ % RED,
        logging.CRITICAL: BOLD_COLOR_SEQ % RED,
    }

    def outputToFile(self):
        return not sys.stdout.isatty() or not sys.stderr.isatty()

    def __init__(self):
        fmt = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
        # Disable the colours when the output is redirected to a file.
        if self.outputToFile():
            super().__init__(fmt)
        else:
            super().__init__("%(color)s" + fmt + self.RESET_COLOR)

    def formatTime(self, record, datefmt=None):
        ct = dt.datetime.fromtimestamp(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%H:%M:%S")
            s = "%s.%03d" % (t, record.msecs)
        return s

    def format(self, record):
        record.color = self.FORMATS[record.levelno]
        record.levelname = record.levelname.lower()
        return super().format(record)


def _excepthook(*args):
    e = traceback.format_exception(*args)
    extra_info = ""
    # If the exception was raised by a subprocess print its
    # stderr / stdout if available.
    if isinstance(args[1], subprocess.CalledProcessError):
        extra_info = args[1].stderr or args[1].stdout
        extra_info = "\n" + extra_info.decode("utf-8")

    if any("[closed]" in repr(h) for h in logger.handlers):
        # In some cases pytest has already closed the logger so use stderr
        # as a fallback.
        print("%s\n%s%s", e[-1], "".join(e), extra_info, file=sys.stderr)
    else:
        logger.critical("%s\n%s%s", e[-1], "".join(e), extra_info)
    sys.exit(1)


_console = logging.StreamHandler()
_console.setFormatter(_PoptorchFormatter())
_console.setLevel(logging.DEBUG)
logger.addHandler(_console)
sys.excepthook = _excepthook
