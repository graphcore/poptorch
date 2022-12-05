# TODO(T71387) Remove LLVM / MLIR completely in follow up patch
set(LLVM_VERSION 15.0.2)

find_package(LLVM ${LLVM_VERSION} EXACT REQUIRED CONFIG)
# Note: MLIR and LLD don't seem to be versioned
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found LLVM: ${LLVM_DIR}")
