cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(poptorch-geometric)

# Run an install command that requires PopTorch, PopArt and Poplar to be in the PATH.
function(run_poppyg_install_command cmd working_directory cmd_name)
  install(CODE
   " execute_process( COMMAND ${cmd} WORKING_DIRECTORY ${working_directory} RESULT_VARIABLE RETVAL OUTPUT_VARIABLE OUTPUT ERROR_VARIABLE OUTPUT)
    if(RETVAL AND NOT RETVAL EQUAL 0)
      message(FATAL_ERROR \"${cmd_name} FAILED: \${OUTPUT}\")
    endif()")
endfunction()

set(INSTALL_POPPYG_PYDIR ${CMAKE_INSTALL_PREFIX}/poptorch_geometric)

add_subdirectory(python)

add_custom_target(poptorch_geometric_wheel
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
  COMMAND python3 ${PROJECT_SOURCE_DIR}/../scripts/generate_poppyg_package.py bdist_wheel --output-dir ${CMAKE_INSTALL_PREFIX}/dist --python-dir ${INSTALL_POPPYG_PYDIR}
)

add_custom_target(poptorch_geometric_sdist
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
  COMMAND python3 ${PROJECT_SOURCE_DIR}/../scripts/generate_poppyg_package.py sdist --output-dir ${CMAKE_INSTALL_PREFIX}/dist --python-dir ${INSTALL_POPPYG_PYDIR}
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md
        DESTINATION .)
