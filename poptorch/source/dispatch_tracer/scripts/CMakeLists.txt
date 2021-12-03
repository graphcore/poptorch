
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/../AtenToMlirInterface.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/../AtenToMlirInterface.hpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/../AtenToMlirDispatch.inc
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate_op_interface.py
     # The file containing the list of operations we are creating an interface for.
     --ops-to-generate "${CMAKE_CURRENT_SOURCE_DIR}/OpSupport.yml"

     # A copy of the upstream native_functions.yml for this version of pytorch.
     --pytorch-base-native-function "${CMAKE_CURRENT_SOURCE_DIR}/native_functions/native_functions.1.${TORCH_MINOR_VERSION}.yml"

     # The file to generate into.
     --gen-cpp-file-path "${CMAKE_CURRENT_BINARY_DIR}/../AtenToMlirInterface.cpp.inc"

      # We create a header file for the methods as well.
     --gen-hpp-file-path "${CMAKE_CURRENT_BINARY_DIR}/../AtenToMlirInterface.hpp.inc"

      # Finally a helper to create a lookup map which maps the Aten symbol onto the function.
     --gen-lookup "${CMAKE_CURRENT_BINARY_DIR}/../AtenToMlirDispatch.inc"
   DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/OpSupport.yml"
)

add_custom_target(generate_supported_ops_interface
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/../AtenToMlirInterface.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/../AtenToMlirInterface.hpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/../AtenToMlirDispatch.inc
)

add_library(generated_supported_ops_interface INTERFACE)
target_include_directories(generated_supported_ops_interface INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/..)
add_dependencies(generated_supported_ops_interface generate_supported_ops_interface)
