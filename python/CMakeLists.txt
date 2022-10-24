include(GNUInstallDirs)
file(GLOB python_files "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

# __init__.py needs to be edited by set_version.py so don't copy it over.
list(REMOVE_ITEM python_files "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py")

run_poptorch_install_command("python3 ${PROJECT_SOURCE_DIR}/scripts/set_version.py --torch-version ${TORCH_VERSION} ${CMAKE_CURRENT_BINARY_DIR}/__init__.py" "${PROJECT_SOURCE_DIR}" "Generate __init__.py")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py DESTINATION "${INSTALL_PYDIR}")
install(FILES ${python_files}  py.typed DESTINATION "${INSTALL_PYDIR}")

# Compile the Pybind11 module using setup.py (Called by generate_python_package.py
run_poptorch_install_command(
  "python3 ${PROJECT_SOURCE_DIR}/scripts/generate_python_package.py install --include-dir ${CMAKE_INSTALL_PREFIX}/include --lib-dir ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} --output-dir ${CMAKE_INSTALL_PREFIX} --python-dir ${INSTALL_PYDIR}" "${PROJECT_SOURCE_DIR}" "poptorch_core.so module compilation")
