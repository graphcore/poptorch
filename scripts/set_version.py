#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import logging
import os

from utils import _utils

logger = logging.getLogger(os.path.basename(__file__))
_utils.set_logger(logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",
                        "-d",
                        action="store_true",
                        help="Print debug messages")
    parser.add_argument("--torch-version", type=str)
    parser.add_argument("output", help="File to create")

    args = parser.parse_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level)
    logger.debug("Args: %s", str(args))

    pkg_info = _utils.PkgInfo.load_from_file(must_exist=False)

    # Copy the content of python/__init__.py and replace the occurrences of
    # @VERSION@ / @SNAPSHOT@ with the actual version / snapshot
    with open(args.output, "w") as f:
        for line in open(
                os.path.join(_utils.sources_dir(), "python", "__init__.py")):
            line = line.replace("@VERSION@", pkg_info.version_long)
            line = line.replace("@SNAPSHOT@", pkg_info.snapshot)
            if args.torch_version:
                line = line.replace("@TORCH_VERSION@", args.torch_version)
            f.write(line)
