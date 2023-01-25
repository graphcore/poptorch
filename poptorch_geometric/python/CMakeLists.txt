include(GNUInstallDirs)
file(GLOB poppyg_python_files "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

# __init__.py needs to be edited by set_version.py so don't copy it over.
list(REMOVE_ITEM poppyg_python_files "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py")

run_poppyg_install_command("python3 ${PROJECT_SOURCE_DIR}/../scripts/set_version.py --input-file ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/__init__.py" "${PROJECT_SOURCE_DIR}" "Generate __init__.py")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py DESTINATION "${INSTALL_POPPYG_PYDIR}")
install(FILES ${poppyg_python_files} py.typed DESTINATION "${INSTALL_POPPYG_PYDIR}")

run_poppyg_install_command("python3 ${PROJECT_SOURCE_DIR}/../scripts/generate_poppyg_package.py install --output-dir ${CMAKE_INSTALL_PREFIX} --python-dir ${INSTALL_POPPYG_PYDIR}" "${PROJECT_SOURCE_DIR}" "")
