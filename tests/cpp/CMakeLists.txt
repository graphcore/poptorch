# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.70 REQUIRED COMPONENTS unit_test_framework)

# Ensure ABI matches that of PyTorch
add_definitions(${TORCH_CXX_FLAGS})

function(add_unit_test name)
  add_executable(${name} ${ARGN})

  target_link_libraries(${name} Boost::unit_test_framework torch poptorch poptorch_logging)

  target_include_directories(${name} PRIVATE
    ${CMAKE_SOURCE_DIR}/poptorch/source/include/)

  add_test(${name} ${name})

endfunction()

add_unit_test(GNNOptimizationsTest GNNOptimizationsTest.cpp)