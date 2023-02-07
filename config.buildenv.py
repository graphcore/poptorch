# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

# IMPORTANT: Keep requirements in sync with ./requirements.txt.

_llvm_version = "13.0.1"

installers.add(
    CondaPackages(
        "ccache=4.3",
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
    installers.add(
        CondaPackages(
            "clang-tools=" + _llvm_version,
            "pylint=2.7.2",
            "yapf=0.27.0",
            # To preserve the comments when updating the schemas
            "ruamel.yaml=0.17.21",
        ))


class DownloadExternalDatasets(Installer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.downloader_path = os.path.join(_utils.sources_dir(), 'scripts',
                                            'download_external_datasets.py')
        if not os.path.exists(self.downloader_path):
            raise RuntimeError(f'Path {self.downloader_path} not exists.')

    def hashString(self):
        with open(self.downloader_path, "r") as f:
            return f.read()

    def install(self, env):
        datasets_path = os.path.join(env.prefix, "external_datasets")
        env.run_commands(f"mkdir {datasets_path}",
                         f"python3 {self.downloader_path} {datasets_path}")


installers.add(DownloadExternalDatasets())

installers.add(PipRequirements("poptorch_geometric/requirements.txt"))
