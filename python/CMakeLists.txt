include(GNUInstallDirs)

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

# Ensure ABI matches that of PyTorch
add_definitions("-D_GLIBCXX_USE_CXX11_ABI=${Torch_USE_CXX11_ABI}")

set(POPTORCH_CORE_SOURCES
  "poptorch.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/LowerToPopart.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/OpBuilder.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PopartCanonicalization.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PoplarExecutable.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/ShapeInference.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/Peephole.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/EliminateListConstructs.cpp"
  "${PROJECT_SOURCE_DIR}/shared/source/Logging.cpp"
  )

pybind11_add_module(poptorch_core SHARED
  ${POPTORCH_CORE_SOURCES}
  )

target_link_libraries(poptorch_core PRIVATE
  ${TORCH_LIBRARIES}
  popart_compiler)

target_include_directories(poptorch_core PRIVATE
  "${PROJECT_SOURCE_DIR}/poptorch/include"
  "${PROJECT_SOURCE_DIR}/shared/include"
  "${PROJECT_SOURCE_DIR}/popart_compiler/include"
  ${SPDLOG_INCLUDE_DIR}
  ${TORCH_INCLUDE_DIRS})

install(TARGETS poptorch_core
  DESTINATION ${INSTALL_PYDIR})

install(FILES __init__.py DESTINATION "${INSTALL_PYDIR}")
