


set(MLIR_TABLEGEN_EXE "${LLVM_TOOLS_BINARY_DIR}/${MLIR_TABLEGEN_EXE}")
set(LLVM_TABLEGEN_EXE "${LLVM_TOOLS_BINARY_DIR}/llvm-tblgen")

message(STATUS "LLVM tablegen exe ${TABLEGEN_EXE}")
message(STATUS "Using MLIR_TABLEGEN_EXE ${MLIR_TABLEGEN_EXE}")

add_mlir_interface(PoptorchInterfaces)

# This helper automatically picks up the header/tablegen files in this dir.
# However it will only look for those specific names. If we add more we will
# need to use the functions it uses internally directly.
add_mlir_dialect(Poptorch poptorch)


# After we've created the normal tablegen stuff we generate some extra headers
# to help glue everything together.


# Pure Tablegen records output as JSON.
set(LLVM_TARGET_DEFINITIONS Poptorch.td)
tablegen(LLVM Poptorch.tblgen.out --dump-json)
add_public_tablegen_target(poptorch_raw_tablegen)

add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/AutogenCompiler.hpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/AutogenCompiler.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/JitToMlirDispatch.hpp.inc

  COMMAND python3 ${SCRIPTS_DIR}/generate_extra_headers.py
    --input-file "${CMAKE_CURRENT_BINARY_DIR}/Poptorch.tblgen.out"
    --interface-header-file "${CMAKE_CURRENT_BINARY_DIR}/AutogenCompiler.hpp.inc"
    --interface-cpp-file "${CMAKE_CURRENT_BINARY_DIR}/AutogenCompiler.cpp.inc"
    --jit-dispatch-table-output-file "${CMAKE_CURRENT_BINARY_DIR}/JitToMlirDispatch.hpp.inc"
  DEPENDS
    poptorch_raw_tablegen
  )

add_custom_target(poptorch_decls
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/AutogenCompiler.hpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/AutogenCompiler.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/JitToMlirDispatch.hpp.inc
)

add_library(generated_poptorch_decls INTERFACE)
target_include_directories(generated_poptorch_decls INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/..)
add_dependencies(generated_poptorch_decls poptorch_decls)
