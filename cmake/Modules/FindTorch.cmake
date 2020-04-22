set(WhatToDoString "Set Torch_DIR, \
something like -DTorch_DIR=/path/to/directory/containing/TorchConfig.cmake/")

execute_process(COMMAND python -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent, end='')"
                OUTPUT_VARIABLE TorchInit_PATH)

FIND_PATH(TorchLibsPath
  HINTS ${TorchInit_PATH}/lib
  NAMES libtorch.so)
IF(NOT TorchLibsPath)
  MESSAGE(FATAL_ERROR "Could not find libtorch.so. ${WhatToDoString}")
ENDIF()
set(TORCH_LIBRARIES "${TorchLibsPath}/libtorch.so")
list(APPEND TORCH_LIBRARIES "${TorchLibsPath}/libtorch_python.so")

# Include dirs
if (EXISTS "${TorchInit_PATH}/include")
  set(TORCH_INCLUDE_DIRS
    ${TorchInit_PATH}/include
    ${TorchInit_PATH}/include/torch/csrc/api/include)
else()
  message(FATAL_ERROR "Could not find ${TorchInit_PATH}/include")
endif()
