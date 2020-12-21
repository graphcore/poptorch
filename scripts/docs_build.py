#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import logging
import os
import shutil
import subprocess
import sys
import zipfile

import sphinx.cmd.build
from utils import _utils

logger = logging.getLogger(os.path.basename(__file__))
_utils.set_logger(logger)


class DocumentationBuilder:
    def __init__(self, pkg_info, install_dir=None):
        self.pkg_info = pkg_info
        self.output_dir = "docs"
        self.output_pdf_dir = os.path.join(self.output_dir, "pdf")
        self.output_html_dir = os.path.join(self.output_dir, "html")
        self.output_guide_dir = os.path.join(self.output_html_dir,
                                             pkg_info.doc_name)
        self.docs_src_dir = os.path.join(_utils.sources_dir(), "docs",
                                         "user_guide")
        self.sphinx_conf_dir = os.path.join(_utils.sources_dir(), "docs",
                                            "common")
        self.title = _utils.get_first_line(
            os.path.join(self.docs_src_dir, "index.rst"))
        self.install_dir = install_dir or "."
        logger.debug("Document title is %s", self.title)

        # -a  write all files (default: only write new and changed files)
        # -E don't use a saved environment, always read all files
        # -n nit-picky mode, warn about all missing references
        # -W turn warnings into errors
        # -j auto: automatically select the appropriate number of threads
        self.common_sphinx_flags = "-a -E -n -W --keep-going -j auto".split(
        ) + ["-c", self.sphinx_conf_dir]

    def assert_poptorch_in_path(self):
        error = None
        try:
            import poptorch  # pylint: disable=unused-import, import-outside-toplevel
        except ImportError as e:
            error = str(e)
            error += ". poptorch must be in your PYTHONPATH to generate the "
            error += "documentation: did you enable your build environment?"
        if error:
            raise ImportError(error)

    def cleanup(self):
        _utils.rmdir_if_exists(self.output_pdf_dir)
        _utils.rmdir_if_exists(self.output_guide_dir)
        os.makedirs(self.output_guide_dir)

    def build_html(self):
        self.assert_poptorch_in_path()
        args = self.common_sphinx_flags + [
            "-b", "html", "-D", f"project=\"{self.title}\"", "-D",
            f"html_title=\"{self.title}\"", "-D",
            f"version=\"v{self.pkg_info.version_long}\"", self.docs_src_dir,
            self.output_guide_dir
        ]
        assert not sphinx.cmd.build.build_main(args), (
            f"The command sphinx-build {' '.join(args)} failed "
            "(See above for details)")

    def package_html(self):
        archive = zipfile.ZipFile(
            os.path.join(self.install_dir, self.pkg_info.html_filename()), "w",
            zipfile.ZIP_DEFLATED)
        excluded_dirs = [".doctrees", "_sources"]
        excluded_files = ["objects.inv", ".buildinfo"]
        for root, _, files in os.walk(self.output_guide_dir):
            if any([root.endswith(ex) for ex in excluded_dirs]):
                continue

            # Remove docs/html/ prefix
            new_root = root.replace(self.output_html_dir,
                                    "")[1:]  # Remove leading '/'

            for file in files:
                if file in excluded_files:
                    continue
                archive.write(os.path.join(root, file),
                              arcname=os.path.join(new_root, file))
        archive.close()
        logger.info("%s was successfully generated",
                    self.pkg_info.html_filename())

    def build_pdf(self):
        self.assert_poptorch_in_path()
        args = self.common_sphinx_flags + [
            "-b", "latex", "-D", f"project=\"{self.pkg_info.doc_name}\"", "-D",
            f"release=\"v{self.pkg_info.version_long}\"", "-D",
            f"version=\"v{self.pkg_info.version_long}\"", self.docs_src_dir,
            self.output_pdf_dir
        ]
        os.environ["DOC_TITLE"] = self.title
        assert not sphinx.cmd.build.build_main(args), (
            f"The command sphinx-build {' '.join(args)} failed "
            "(See above for details)")
        subprocess.check_output(["make", "LATEXMKOPTS=\"-silent\""],
                                cwd=self.output_pdf_dir)
        shutil.copyfile(
            os.path.join(self.output_pdf_dir, "doc.pdf"),
            os.path.join(self.install_dir, self.pkg_info.pdf_filename()))
        logger.info("%s was successfully generated",
                    self.pkg_info.pdf_filename())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pdf",
                        action="store_true",
                        help="Do not generate the PDF documentation")
    parser.add_argument("--no-html",
                        action="store_true",
                        help="Do not generate the HTML documentation")
    parser.add_argument("--debug",
                        "-d",
                        action="store_true",
                        help="Print debug messages")
    parser.add_argument("--add-to-sys-path", help="Path to add to sys.path")
    parser.add_argument("--install-dir",
                        help="Copy generated files to that folder")

    args = parser.parse_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level)
    logger.debug("Args: %s", str(args))

    if args.add_to_sys_path:
        for path in args.add_to_sys_path.split(";"):
            logger.debug("Adding %s", path)
            sys.path.insert(0, path)

    builder = DocumentationBuilder(
        _utils.PkgInfo.load_from_file(must_exist=False), args.install_dir)
    if not args.no_pdf:
        builder.build_pdf()

    if not args.no_html:
        builder.build_html()
        builder.package_html()
