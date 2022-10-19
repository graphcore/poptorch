# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

set(TARGET_NAME poptorch-opt)

add_executable(${TARGET_NAME}
  main.cpp
)

target_link_libraries(${TARGET_NAME}
  PRIVATE
  LLVMSupport
  MLIRFuncDialect
  MLIRIR
  MLIROptLib

  poptorch_mlir_passes
  poptorch_compiler
  poptorch_dialect
  lower_to_poplar
)

set_target_properties(${TARGET_NAME} PROPERTIES
  CXX_STANDARD 17
  CMAKE_CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)
