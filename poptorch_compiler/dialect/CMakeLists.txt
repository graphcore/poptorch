
add_subdirectory(include)

# Enable exception handling in MLIR, for example if the input arguments are
# invalid for determining the output shape.
set(LLVM_REQUIRES_EH ON)

# add_mlir_dialect_library ignores the INTERFACE_INCLUDE_DIRECTORIES
# when calling the compile command, so we call include_directories
# here to correctly include poptorch_logging.
# (We still attach it to LINK_LIBS so that dependants link against
# poptorch_logging)
include_directories(PRIVATE
    $<TARGET_PROPERTY:poptorch_logging,INTERFACE_INCLUDE_DIRECTORIES>)

add_mlir_dialect_library(poptorch_dialect
    PoptorchDialect.cpp
    ops/NormOpsHelpers.cpp
    ops/Poptorch.cpp
  DEPENDS
    MLIRPoptorchIncGen
    MLIRPoptorchInterfacesIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    generated_poptorch_decls
    poptorch_logging
)

remove_shared_llvm(poptorch_dialect)
