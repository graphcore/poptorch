# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import datetime as dt
import logging
import os
import sys
import subprocess
import traceback

# Create a poptorch logger which outputs to the console INFO messages and above
logger = logging.getLogger("poptorch::python")
log_level = os.environ.get("POPTORCH_LOG_LEVEL", "INFO")
if log_level in ["DEBUG", "TRACE", "TRACE_ALL"]:
    logger.setLevel(logging.DEBUG)
elif log_level == "INFO":
    logger.setLevel(logging.INFO)
elif log_level == "WARN":
    logger.setLevel(logging.WARNING)
elif log_level == "ERR":
    logger.setLevel(logging.ERROR)
elif log_level == "OFF":
    # We don't use CRITICAL so that's
    # essentially OFF
    logger.setLevel(logging.CRITICAL)
else:
    raise RuntimeError(f"Unknown POPTORCH_LOG_LEVEL {log_level}")


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
