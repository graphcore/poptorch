
add_custom_target(generate_supported_ops_interface DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/OpSupport.yml")

add_custom_command(TARGET generate_supported_ops_interface
                COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate_op_interface.py

                            # The file containing the list of operations we are creating an interface for.
                           --ops-to-generate "${CMAKE_CURRENT_SOURCE_DIR}/OpSupport.yml"

                           # A copy of the upstream native_functions.yml for this version of pytorch.
                           --pytorch-base-native-function "${CMAKE_CURRENT_SOURCE_DIR}/native_functions/native_functions.1.${TORCH_MINOR_VERSION}.yml"

                           # The file to generate into.
                           --gen-cpp-file-path "${CMAKE_CURRENT_SOURCE_DIR}/../dispatchers/AtenToMlirInterface.cpp.inc"

                            # We create a header file for the methods as well.
                           --gen-hpp-file-path "${CMAKE_CURRENT_SOURCE_DIR}/../dispatchers/AtenToMlirInterface.hpp.inc"

                            # Finally a helper to create a lookup map which maps the Aten symbol onto the function.
                           --gen-lookup "${CMAKE_CURRENT_SOURCE_DIR}/../dispatchers/AtenToMlirDispatch.inc"
)