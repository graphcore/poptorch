
add_subdirectory(include)

add_mlir_dialect_library(poptorch_dialect
    PopTorchDialect.cpp
  DEPENDS
    MLIRPoptorchIncGen
    MLIRPoptorchInterfacesIncGen

  LINK_LIBS PUBLIC
    MLIRIR
)

remove_shared_llvm(poptorch_dialect)

# Add the api generator script as a dependency so it gets run when the tablegen changes.
add_dependencies(poptorch_dialect poptorch_decls)
