# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
installers.add(
    CondaPackages(
        "ccache=3.7.9",
        "cmake=3.18.2",
        "hunspell=1.7.0",
        "latexmk=4.55",
        "make=4.3",
        "ninja=1.10.2",
        "pybind11=2.6.1",
        "pytest=6.2.1",
        "zip=3.0",
        "spdlog=1.8.0",
        "wheel=0.34.2",
    ))

installers.add(PipRequirements("requirements.txt"))

if config.install_linters:
    installers.add(
        CondaPackages("yapf=0.27.0", "cpplint=1.4.4", "pylint=2.5.3",
                      "clang-tools=9.0.0", "python-clang=9.0.0",
                      "pyyaml=5.3.1"))
