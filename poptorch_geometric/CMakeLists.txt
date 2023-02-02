cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(poptorch-geometric)

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
