file(GLOB_RECURSE poptorch_compiler_public_headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp*")

add_definitions(${LLVM_DEFINITIONS})

add_library(poptorch_compiler SHARED
  PoptorchCompiler.cpp
  IMLIRCompiler.cpp
  Ops.cpp
  PytorchBridgeUtils.cpp
  IpuSession.cpp
)

# This is needed as PopIR will look for its symbols in our MLIR static libs when it links against PopTorch.
target_compile_options(poptorch_compiler PRIVATE -fexceptions -Wl,--exclude-libs,ALL)

target_link_libraries(poptorch_compiler
  PUBLIC
    poptorch_dialect
  PRIVATE
    MLIRIR
    MLIRSupport
    LLVMSupport
    poptorch_mlir_passes
)

set_property(TARGET poptorch_compiler PROPERTY CXX_STANDARD 17)

set_target_properties(poptorch_compiler PROPERTIES
  PUBLIC_HEADER "${poptorch_compiler_public_headers}")

target_include_directories(poptorch_compiler
                            PUBLIC
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                            $<INSTALL_INTERFACE:include>
                            )
install(TARGETS poptorch_compiler
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pytorch_bridge
)
