
file(GLOB_RECURSE poptorch_compiler_public_headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp*")

if (POPTORCH_BUILD_MLIR_COMPILER)
  add_definitions(${LLVM_DEFINITIONS})

  add_library(poptorch_compiler SHARED
    PoptorchCompiler.cpp
    IMLIRCompiler.cpp
    MLIREagerCompiler.cpp
    MLIRStaticGraphCompiler.cpp
    PoplarExecutorWrapper.cpp
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

    lower_to_poplar
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
else()
  # Even though we don't build the poptorch compiler, the dispatch_tracer in
  # Core still depends on some of the headers.
  add_library(poptorch_compiler INTERFACE)
  target_include_directories(poptorch_compiler
                             INTERFACE
                             $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                             $<INSTALL_INTERFACE:include>
                             )
  install(FILES ${poptorch_compiler_public_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pytorch_bridge)
endif()
