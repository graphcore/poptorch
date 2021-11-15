
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


# Run the script to generate the C++.
add_dependencies(dispatch_tracer generate_supported_ops_interface)

target_link_libraries(dispatch_tracer
  PUBLIC
    torch
  PRIVATE
  poptorch_logging)

# Add the compiler if we are to build that.
if (POPTORCH_BUILD_MLIR_COMPILER)
  target_link_libraries(dispatch_tracer PRIVATE poptorch_compiler)
endif()

set_property(TARGET dispatch_tracer PROPERTY CXX_STANDARD 17)

target_include_directories(dispatch_tracer PRIVATE
  "${PROJECT_SOURCE_DIR}/poptorch/include"
  "${PROJECT_SOURCE_DIR}/popart_compiler/include"
  # Include the codegen.
  "${PROJECT_SOURCE_DIR}/poptorch_compiler/pytorch_bridge/include"
)