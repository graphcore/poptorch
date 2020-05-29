include(GNUInstallDirs)

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

# Ensure ABI matches that of PyTorch
add_definitions("-D_GLIBCXX_USE_CXX11_ABI=${Torch_USE_CXX11_ABI}")

set(POPTORCH_CORE_SOURCES
  "poptorch.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/error.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/LowerToPopart.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/OpBuilder.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PopartCanonicalization.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PoplarExecutable.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/ShapeInference.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/Peephole.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/EliminateListConstructs.cpp"
  )

set(POPTORCH_CORE_SHARED_SOURCES
  "${PROJECT_SOURCE_DIR}/shared/source/Logging.cpp"
  )

if(NOT ${Torch_USE_CXX11_ABI})
  list(APPEND POPTORCH_CORE_SOURCES
       ${POPTORCH_CORE_SHARED_SOURCES})
endif()


pybind11_add_module(poptorch_core SHARED
  ${POPTORCH_CORE_SOURCES}
  )

target_link_libraries(poptorch_core PRIVATE
  ${TORCH_LIBRARIES}
  ${POPART_LIB}
  ${POPLAR_LIB}
  ${POPUTIL_LIB}
  popart_compiler)

target_include_directories(poptorch_core PRIVATE
  "${PROJECT_SOURCE_DIR}/poptorch/include"
  "${PROJECT_SOURCE_DIR}/shared/include"
  "${PROJECT_SOURCE_DIR}/popart_compiler/include"
  ${POPART_INCLUDE_DIR}
  ${POPLAR_INCLUDE_DIR}
  ${POPLIBS_INCLUDE_DIR}
  ${SPDLOG_INCLUDE_DIR}
  ${TORCH_INCLUDE_DIRS})

install(TARGETS poptorch_core
  DESTINATION ${INSTALL_PYDIR})

install(FILES __init__.py DESTINATION "${INSTALL_PYDIR}")
