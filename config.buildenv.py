# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

# IMPORTANT: Keep requirements in sync with ./requirements.txt.

_llvm_version = "12.0.1"

installers.add(
    CondaPackages(
        "ccache=4.3",
        "cmake=3.18.2",
        "lld=" + _llvm_version,
        "llvmdev=" + _llvm_version,
        "make=4.3",
        "mlir=" + _llvm_version,
        "ninja=1.10.2",
        "pybind11=2.6.1",
        "pytest=6.2.1",
        "pyyaml=5.3.1",
        "setuptools=58.0.4",
        "spdlog=1.8.0",
        "typing_extensions=4.1.1",
        "wheel=0.34.2",
        "zip=3.0",
    ))

if not config.is_aarch64:
    # These packages don't exist on AArch64 but they're only needed to
    # build the documentation
    installers.add(CondaPackages(
        "hunspell=1.7.0",
        "latexmk=4.55",
    ))

installers.add(PipRequirements("requirements.txt"))

if config.install_linters:
    installers.add(
        CondaPackages(
            "clang-tools=" + _llvm_version,
            "cpplint=1.4.4",
            "pylint=2.5.3",
            "python-clang=" + _llvm_version,
            "yapf=0.27.0",
        ))
