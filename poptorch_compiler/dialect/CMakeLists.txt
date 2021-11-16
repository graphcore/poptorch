
add_subdirectory(include)

add_mlir_dialect_library(poptorch_dialect
    PoptorchDialect.cpp
  DEPENDS
    MLIRPoptorchIncGen
    MLIRPoptorchInterfacesIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    generated_poptorch_decls
)

remove_shared_llvm(poptorch_dialect)
