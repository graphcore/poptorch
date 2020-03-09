set(WhatToDoString "Set Torch_DIR, \
something like -DTorch_DIR=/path/to/directory/containing/TorchConfig.cmake/")

execute_process(COMMAND python -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent, end='')"
                OUTPUT_VARIABLE TorchInit_PATH)

FIND_PATH(TorchConfig_DIR 
  HINTS ${TorchInit_PATH}/share/cmake/Torch
  NAMES TorchConfig.cmake)
IF(NOT TorchConfig_DIR)
  MESSAGE(FATAL_ERROR "Could find TorchConfig.cmake. ${WhatToDoString}")
ENDIF()
include(${TorchConfig_DIR}/TorchConfig.cmake)
