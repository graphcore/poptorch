
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Generates the C++ interface to add each pytorch op.
add_subdirectory(scripts)


# This can be built without the MLIR component.
if (POPTORCH_BUILD_MLIR_COMPILER)
  set(MLIR_SOURCE_FILE "dispatchers/MlirDispatch.cpp")
endif()


add_library(dispatch_tracer STATIC
        RegisterAtenOverloads.cpp
        CommonHelperFunctions.cpp
        dispatchers/JitDispatch.cpp

        # The mlir dispatcher or null.
        ${MLIR_SOURCE_FILE}

        dispatchers/CompilerDispatchTable.cpp
        ValueMapper.cpp
)

target_link_libraries(dispatch_tracer
  PUBLIC
    torch
  PRIVATE
  poptorch_core_headers
  poptorch_logging
  poptorch_compiler
  popart_compiler
  generated_supported_ops_interface
  )

# Add the compiler if we are to build that.
if (POPTORCH_BUILD_MLIR_COMPILER)
  target_link_libraries(dispatch_tracer PRIVATE poptorch_compiler)
endif()

set_property(TARGET dispatch_tracer PROPERTY CXX_STANDARD 17)
