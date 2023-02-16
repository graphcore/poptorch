# PopTorch and PopTorch Geometric.

## PopTorch - PyTorch integration for the Graphcore IPU

PopTorch is a set of extensions for PyTorch enabling models
to be trained, evaluated and used on the Graphcore IPU.

More information can be found in the [PopTorch User Guide](https://docs.graphcore.ai/projects/poptorch-user-guide/).

## PopTorch Geometric - PyTorch Geometric integration for the Graphcore IPU

PopTorch Geometric is a set of extensions for PyTorch Geometric, enabling Graph
Neural Network models to be trained, evaluated and used on the Graphcore IPU.
PopTorch Geometric depends on the functionality provided by PopTorch.

More information can be found in the [PopTorch User Guide](https://docs.graphcore.ai/projects/poptorch-geometric-user-guide).


## Prerequisites

These instructions assume you are building PopTorch and PopTorch Geometric on Ubuntu 20.04.

To install and run PopTorch and PopTorch Geometric you will need:

- Python 3.8
- pip3 >= 18.1
- The Poplar SDK

```sh
sudo apt install -y python3 python3-pip
```

To build PopTorch and PopTorch Geometric from sources you will need all of the above and:

- git
- curl
- g++

```sh
sudo apt install -y git curl g++
```

To build the documentation you will also need LaTeX:

```sh
sudo apt install -y texlive-full
```

## Install the Poplar SDK

The Poplar SDK can be downloaded from: https://www.graphcore.ai/downloads.

Set the following environment variable to point to the installed Poplar SDK:

```sh
export SDK_PATH=/path/to/poplar_sdk-ubuntu_20_04*
```

PopTorch must be built against a compatible version of the SDK. For example, the "sdk-release-3.2" branch of PopTorch must be built against Poplar SDK 3.2.

## Installation

Make sure `pip3` is up to date (You need `pip3 >= 18.1`):

```sh
pip3 install -U pip --user
```

Install the PopTorch wheel (Torch will automatically be installed in the
process):

```sh
pip3 install ${SDK_PATH}/poptorch-*.whl
```

Once the PopTorch wheel has been installed, PopTorch Geometric wheel can be
installed if needed (PyTorch Geometric will automatically be installed in
the process):

```sh
pip3 install ${SDK_PATH}/poptorch_geometric-*.whl
```

## Usage

The PopTorch wheel doesn't include the PopART and Poplar binaries, so you need to make sure they are in your path before loading PopTorch or PopTorch Geometric.
This is done by sourcing their respective `enable.sh` scripts:

```sh
. ${SDK_PATH}/poplar-ubuntu_20_04*/enable.sh
. ${SDK_PATH}/popart-ubuntu_20_04*/enable.sh
```

You can check everything is in order by running:

```sh
python3 -c "import poptorch;print(poptorch.__version__)"
```

And similarly for PopTorch Geometric:

```sh
python3 -c "import poptorch_geometric;print(poptorch_geometric.__version__)"
```

More information can be found in the [PopTorch User Guide](https://docs.graphcore.ai/projects/poptorch-user-guide/)

## Build instructions

Like [PyTorch](https://pytorch.org/) we use [Anaconda](https://anaconda.org/anaconda/conda) as build environment manager.

1. Clone the PopTorch repository

```sh
git clone https://github.com/graphcore/poptorch.git
```

2. Create a folder for your build

```sh
mkdir build
cd build
```

3. Create a build environment and install the dependencies.

```sh
../poptorch/scripts/create_buildenv.py
```

4. Activate the build environment

```sh
. activate_buildenv.sh
```

5. Configure the build

```sh
cmake ../poptorch -DSDK_DIR=${SDK_PATH} -GNinja
```

By default, PopTorch will be built in release mode. To build in debug mode add `-DCMAKE_BUILD_TYPE=Debug`.

To build the documentation, add `-DBUILD_DOCS=ON`. The HTML and PDF documentation will be generated in `docs/`.

6. Compile the PopTorch and PopTorch Geometric libraries

```sh
ninja install
```

If you're only going to use PopTorch or PopTorch Geometric for development purposes then you can stop here.
Source the enable script in the PopTorch build folder and you can start using PopTorch:

```sh
. enable.sh
python3 -c "import poptorch;print(poptorch.__version__)"
```

Similarly for PopTorch Geometric:
```sh
. enable.sh
python3 -c "import poptorch_geometric;print(poptorch_geometric.__version__)"
```

7. (Optional) Build the PopTorch wheel

```sh
ninja poptorch_wheel
```

The wheel will be created in `install/dist`.

8. (Optional) Build the PopTorch Geometric wheel

```sh
ninja poptorch_geometric_wheel
```

The wheel will be created in `install/dist`.

### Run the tests

To run the tests:

```sh
# Run all the tests, print the output only on failure, run 80 tests in parallel
./test.sh -j80
# PopTorch has 3 test labels: examples, short, long. To run all the tests except the long ones:
./test.sh -j80 -LE long
# To run only the short tests
./test.sh -j80 -L short
# Filter the tests by name using -R
./test.sh -j80 -R half_
# For more information:
./test.sh --help
```

Note: If you run the tests in parallel, make sure to tell PopTorch to wait for an IPU to become available if they are all in use:

```sh
export POPTORCH_WAIT_FOR_IPU=1
```

Tests can also be run individually using `pytest`:

```sh
. enable.sh
python3 -m pytest ../poptorch/tests/options_test.py
# add -s to get the whole output
# -k to filter the tests by name
python3 -m pytest ../poptorch/tests/options_test.py -s -k popart
```

Tests specific for Graph Neural Networks are located in `tests/gnn/` subdirectory:

```sh
. enable.sh
python3 -m pytest ../poptorch/tests/gnn/test_basic_gnn.py
# add -s to get the whole output
# -k to filter the tests by name
python3 -m pytest ../poptorch/tests/gnn/test_basic_gnn.py -s -k GraphSAGE
```

## Feedback / issues

Please create issues [here](https://github.com/graphcore/poptorch/issues)
