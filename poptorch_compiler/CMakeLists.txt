cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(poptorch_compiler)

# We need -fpic if we are static
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found LLVM in dir : ${LLVM_DIR}")

function(remove_shared_llvm target)
  foreach(prop INTERFACE_LINK_LIBRARIES LINK_LIBRARIES)
    get_target_property(libs ${target} ${prop})
    list(REMOVE_ITEM libs LLVM)
    set_target_properties(${target} PROPERTIES ${prop} "${libs}")
  endforeach()
endfunction()

foreach(lib ${MLIR_EXPORTED_TARGETS})
  remove_shared_llvm(${lib})
endforeach()

find_package(poplar REQUIRED)
find_package(popir REQUIRED)
find_package(poplibs REQUIRED)

# Add LLVM to the module path. Needed to find the "include" code below.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")


add_definitions(${LLVM_DEFINITIONS})

include(AddLLVM)
include(AddMLIR)
include(TableGen)
include(HandleLLVMOptions)


message(STATUS LLVM_LIBRARY_DIRS "${LLVM_LIBRARY_DIRS}")
message(STATUS LLVM_INCLUDE_DIRS "${LLVM_INCLUDE_DIRS}")
message(STATUS MLIR_INCLUDE_DIRS "${LLVM_INCLUDE_DIRS}")

# LLVM library includes.
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

# Use lld instead of gold for the poptorch_compiler (To avoid issues when linking against LLVM)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=lld")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fuse-ld=lld")

# Setup for our sources + generated tablegen includes
# Add the root project.
include_directories("${PROJECT_SOURCE_DIR}/include")

# Add the tablegen.
# Not much we can do about warnings as errors in the tablegen.
include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/dialect/include/)


# Add the dialect to include path.
include_directories("${PROJECT_SOURCE_DIR}/dialect/include/")

set(SCRIPTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/scripts/")
set(BRIDGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pytorch_bridge/")

# Build the dialect.
add_subdirectory(dialect)


# Add lower to poplar to include path.
include_directories("${PROJECT_SOURCE_DIR}/lower_to_poplar/include/")

# Include the PyTorch interface
include_directories("${PROJECT_SOURCE_DIR}/pytorch_bridge/include/")

# Include the generic passes
include_directories("${PROJECT_SOURCE_DIR}/transforms/include/")

# Build the mlir passes
add_subdirectory(transforms)

# Build the poplar lowering code.
add_subdirectory(lower_to_poplar)

# Add the main LIB.
add_subdirectory(pytorch_bridge)

# LLVM can lead to a few link problems.
message(STATUS "Checking LLVM link libs:\n")
mlir_check_all_link_libraries(poptorch_compiler)
mlir_check_all_link_libraries(lower_to_poplar)
mlir_check_all_link_libraries(poptorch_dialect)

install(TARGETS poptorch_compiler
  DESTINATION ${INSTALL_PYDIR})
