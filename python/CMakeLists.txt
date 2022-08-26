include(GNUInstallDirs)

find_package(pybind11 REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# Ensure ABI matches that of PyTorch
add_definitions(${TORCH_CXX_FLAGS})

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
# Workaround for clang / pybind / c++17
# https://github.com/pybind/pybind11/issues/1818
add_compile_options(-fsized-deallocation)
endif()


# Same as pybind11_add_module but without stripping the symbols and setting the visibility to hidden.
# Source: https://pybind11.readthedocs.io/en/stable/compiling.html#advanced-interface-library-targets
#
# If the symbols are stripped then error messages will only contain symbol
# addresses instead of human readable names.
add_library(poptorch_core MODULE
  "poptorch.cpp"
  )
pybind11_extension(poptorch_core)
# skip pybind11_strip(target)

target_link_libraries(poptorch_core PRIVATE
  pybind11::module
  pybind11::lto
  poptorch
  popart_compiler
  torch
  ${TORCH_PYTHON_LIBRARY}
  poptorch_err
  poptorch_logging
  dispatch_tracer
  poptorch_internal_headers
  stdc++fs
)

target_include_directories(poptorch_core PRIVATE ${PROJECT_SOURCE_DIR}/poptorch_compiler/pytorch_bridge/include)

install(TARGETS poptorch_core
  DESTINATION ${INSTALL_PYDIR})

file(GLOB python_files "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

# __init__.py needs to be edited by set_version.py so don't copy it over.
list(REMOVE_ITEM python_files "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py")

run_poptorch_install_command("python3 ${PROJECT_SOURCE_DIR}/scripts/set_version.py --torch-version ${TORCH_VERSION} ${CMAKE_CURRENT_BINARY_DIR}/__init__.py" "${PROJECT_SOURCE_DIR}" "Generate __init__.py")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py DESTINATION "${INSTALL_PYDIR}")
install(FILES ${python_files}  py.typed DESTINATION "${INSTALL_PYDIR}")
