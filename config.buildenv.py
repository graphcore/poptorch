# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

_llvm_version = "12.0.1"

installers.add(
    CondaPackages(
        "ccache=3.7.9",
        "cmake=3.18.2",
        "hunspell=1.7.0",
        "latexmk=4.55",
        "lld==" + _llvm_version,
        "llvmdev==" + _llvm_version,
        "make=4.3",
        "mlir==" + _llvm_version,
        "ninja=1.10.2",
        "pybind11=2.6.1",
        "pytest=6.2.1",
        "spdlog=1.8.0",
        "wheel=0.34.2",
        "zip=3.0",
    ))

installers.add(PipRequirements("requirements.txt"))

if config.install_linters:
    installers.add(
        CondaPackages(
            "clang-tools=" + _llvm_version,
            "cpplint=1.4.4",
            "pylint=2.5.3",
            "python-clang=" + _llvm_version,
            "pyyaml=5.3.1",
            "yapf=0.27.0",
        ))
