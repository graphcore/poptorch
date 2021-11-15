
add_definitions(${LLVM_DEFINITIONS})

add_library(poptorch_compiler SHARED
  Bridge.cpp
  CompilerImpl.cpp
  Executor.cpp
  Ops.cpp
)

# This is needed as PopIR will look for its symbols in our MLIR static libs when it links against PopTorch.
target_compile_options(poptorch_compiler PRIVATE -fno-rtti -fexceptions -Wl,--exclude-libs,ALL)

target_include_directories(poptorch_compiler PUBLIC include)
target_link_libraries(poptorch_compiler
  PRIVATE
  MLIRIR
  MLIRSupport
  LLVMSupport

  poptorch_dialect
  lower_to_poplar
  poprithms
)

set_property(TARGET poptorch_compiler PROPERTY CXX_STANDARD 17)
