set(WhatToDoString "Set Torch_DIR, \
something like -DTorch_DIR=/path/to/directory/containing/TorchConfig.cmake/")

execute_process(COMMAND python -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent, end='')"
                OUTPUT_VARIABLE TorchInit_PATH)

# PyTorch may be compiled with _GLIBCXX_USE_CXX11_ABI=0
execute_process(COMMAND
python -c "import torch; print('1' if torch.compiled_with_cxx11_abi() else '0', end='')"
                OUTPUT_VARIABLE Torch_USE_CXX11_ABI)

find_library(LibTorch torch ${TorchInit_PATH}/lib)
if (NOT LibTorch)
  message(FATAL_ERROR "Could not find shared library for torch.")
endif()

find_library(LibTorchPython torch_python ${TorchInit_PATH}/lib)
if (NOT LibTorchPython)
  message(FATAL_ERROR "Could not find shared library for torch_python.")
endif()

set(TORCH_LIBRARIES "${LibTorch};${LibTorchPython}")

# Include dirs
if (EXISTS "${TorchInit_PATH}/include")
  set(TORCH_INCLUDE_DIRS
    ${TorchInit_PATH}/include
    ${TorchInit_PATH}/include/torch/csrc/api/include)
else()
  message(FATAL_ERROR "Could not find ${TorchInit_PATH}/include")
endif()
