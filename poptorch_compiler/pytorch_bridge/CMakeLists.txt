
if (POPTORCH_BUILD_MLIR_COMPILER)
  add_definitions(${LLVM_DEFINITIONS})

  add_library(poptorch_compiler SHARED
    PoptorchCompiler.cpp
    PoptorchCompilerImpl.cpp
    PoplarExecutorWrapper.cpp
    Ops.cpp
    PytorchBridgeUtils.cpp
  )

# This is needed as PopIR will look for its symbols in our MLIR static libs when it links against PopTorch.
  target_compile_options(poptorch_compiler PRIVATE -fno-rtti -fexceptions -Wl,--exclude-libs,ALL)

  target_include_directories(poptorch_compiler PUBLIC include)
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
else()
  # Even though we don't build the poptorch compiler, the dispatch_tracer in
  # Core still depends on some of the headers.
  add_library(poptorch_compiler INTERFACE)
  target_include_directories(poptorch_compiler INTERFACE include)
endif()
