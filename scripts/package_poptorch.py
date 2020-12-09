#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import shutil
import logging
import glob
import os

from utils import _utils

logger = logging.getLogger(os.path.basename(__file__))
_utils.set_logger(logger)


class PoptorchPackager:
    def __init__(self, pkg_info):
        self.pkg_info = pkg_info

    def create_yml_file(self, include_documentation, output_dir,
                        wheel_filename):
        with open(os.path.join(output_dir, self.pkg_info.prodinfo_filename()),
                  "w") as f:
            f.write("id : %s\n" % self.pkg_info.project_name)
            f.write("type : tool\n")
            f.write("title: PyTorch extensions for IPU\n")
            f.write("version: %s\n" % self.pkg_info.version_long)
            f.write("source_id: %s\n" % self.pkg_info.snapshot)
            f.write(
                "repo: git@phabricator.sourcevertex.net:diffusion/POPTORCH/poptorch.git\n"  # pylint: disable=line-too-long
            )
            f.write("elements:\n")
            f.write("  - name: PopTorch framework (%s)\n" %
                    self.pkg_info.os_type)
            f.write("    file: %s\n" % wheel_filename)
            f.write("    type: %s_installer\n" % self.pkg_info.package_os_type)

            if include_documentation:
                f.write("  - name: PopTorch user guide (PDF)\n")
                f.write("    file: %s\n" % self.pkg_info.pdf_filename())
                f.write("    type: pdf\n")

                f.write("  - name: PopTorch user guide (HTML)\n")
                f.write("    file: %s\n" % self.pkg_info.html_filename())
                f.write("    type: html_zip\n")

    def create_package(self,
                       include_documentation,
                       output_dir,
                       keep_output_dir=False):
        if not keep_output_dir:
            _utils.rmdir_if_exists(output_dir)

        os.makedirs(output_dir, exist_ok=keep_output_dir)

        wheels = glob.glob("dist/*.whl")
        assert wheels, "Couldn't find any whl file in dist/"
        assert len(wheels) == 1, f"Found more than one wheel {wheels}"
        src_wheel = wheels[0]
        # Add the snapshot and package_os_type to the wheel name
        dst_wheel = os.path.basename(src_wheel).replace(
            f"-{self.pkg_info.version}-",
            f"-{self.pkg_info.version}+{self.pkg_info.snapshot}+"
            f"{self.pkg_info.os_type}-")
        shutil.copy(src_wheel, os.path.join(output_dir, dst_wheel))

        if include_documentation:
            shutil.copy(self.pkg_info.pdf_filename(), output_dir)
            shutil.copy(self.pkg_info.html_filename(), output_dir)

        self.create_yml_file(include_documentation, output_dir, dst_wheel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-documentation",
                        action="store_true",
                        help="Package the documentation too")
    # TODO(T27444): Temporarily needed to avoid deleting PopART artifacts...
    parser.add_argument("--keep-output-dir",
                        action="store_true",
                        help="Don't delete the output-dir if it exists")
    parser.add_argument("--output-dir",
                        type=str,
                        default="pkg",
                        help="Output directory")
    parser.add_argument("--source-dir",
                        type=str,
                        help="path to CMake install folder")
    parser.add_argument("--debug",
                        "-d",
                        action="store_true",
                        help="Print debug messages")
    # TODO(T27444): At the moment we can't rely on BUILD_DOCS in CMake
    # to know if the doc should be generated, so instead package the
    # documentation if the files exist
    parser.add_argument("--include-documentation-if-exists",
                        action="store_true",
                        help="Package the documentation too")

    args = parser.parse_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level)

    if args.source_dir:
        os.chdir(args.source_dir)
    packager = PoptorchPackager(_utils.JenkinsPkgInfo())
    if args.include_documentation_if_exists:
        args.include_documentation = os.path.exists(
            packager.pkg_info.pdf_filename())

    packager.create_package(args.include_documentation, args.output_dir,
                            args.keep_output_dir)
