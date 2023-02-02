include(GNUInstallDirs)
file(GLOB poppyg_python_files "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

# __init__.py needs to be edited by set_version.py so don't copy it over.
list(REMOVE_ITEM poppyg_python_files "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py")

install(CODE
   " execute_process(
        COMMAND python3 ${PROJECT_SOURCE_DIR}/../scripts/set_version.py --input-file ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE RETVAL OUTPUT_VARIABLE OUTPUT ERROR_VARIABLE OUTPUT)
    if(RETVAL AND NOT RETVAL EQUAL 0)
      message(FATAL_ERROR \"set_version.py FAILED: \${OUTPUT}\")
    endif()")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py DESTINATION "${INSTALL_POPPYG_PYDIR}")
install(FILES ${poppyg_python_files} py.typed DESTINATION "${INSTALL_POPPYG_PYDIR}")

install(CODE
   " execute_process(
        COMMAND python3 ${PROJECT_SOURCE_DIR}/../scripts/generate_poppyg_package.py install --output-dir ${CMAKE_INSTALL_PREFIX} --python-dir ${INSTALL_POPPYG_PYDIR}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE RETVAL OUTPUT_VARIABLE OUTPUT ERROR_VARIABLE OUTPUT)
    if(RETVAL AND NOT RETVAL EQUAL 0)
      message(FATAL_ERROR \"generate_poppyg_package.py FAILED: \${OUTPUT}\")
    endif()")
