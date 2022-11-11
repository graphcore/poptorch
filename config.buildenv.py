# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

# IMPORTANT: Keep requirements in sync with ./requirements.txt.

config.setDefault(build_llvm=True)


class BuildLLVM(Installer):
    def __init__(self):
        llvm_version = self._get_llvm_version()
        self.url = ("https://github.com/llvm/llvm-project/archive/refs/tags/"
                    f"llvmorg-{llvm_version}.tar.gz")

    def _get_llvm_version(self):
        import re
        with open("poptorch_compiler/CMakeLists.txt") as f:
            for line in f:
                m = re.match(".*LLVM_VERSION ([0-9.]+)\)", line)
                if m:
                    return m.group(1)
        raise ValueError("LLVM_VERSION not found")

    def hashString(self):
        return self.url

    def install(self, env):
        os.makedirs(os.path.join("llvm", "build"))
        env.run_commands(
            "cd llvm",
            f"curl -sSL {self.url} | tar zx --strip-components=1",
            "cd build",
            "export CXX=g++",
            "export CC=gcc",
            f"cmake ../llvm -GNinja \
                    -DCMAKE_BUILD_TYPE=Release \
                    -DLLVM_TARGETS_TO_BUILD:STRING=host \
                    -DLLVM_INCLUDE_GO_TESTS:BOOL=OFF \
                    -DLLVM_ENABLE_LIBCXX:BOOL=ON \
                    -DLLVM_ENABLE_TERMINFO:BOOL=OFF \
                    -DLLVM_INSTALL_UTILS:BOOL=True \
                    -DLLVM_ENABLE_RTTI:BOOL=True \
                    -DLLVM_OPTIMIZED_TABLEGEN:BOOL=ON \
                    -DLLVM_ENABLE_PROJECTS:STRING=\"clang;lld;clang-tools-extra;mlir\" \
                    -DCMAKE_INSTALL_PREFIX={env.prefix}",
            "ninja",
            "ninja install",
        )
        env.rmdir_if_exists("llvm")


installers.add(
    CondaPackages(
        "cmake=3.18.2",
        "libstdcxx-ng=11.2.0",
        "make=4.3",
        "ninja=1.10.2",
        "pybind11=2.6.1",
        "pytest=6.2.5",
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
    installers.add(CondaPackages(
        "pylint=2.7.2",
        "yapf=0.27.0",
    ))

if config.build_llvm:
    installers.add(BuildLLVM())
else:
    # ccache pulls in zstd which has a broken CMake config
    # which causes the LLVM build to fail.
    installers.add(CondaPackages("ccache=4.3", ))
