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

add_library(poptorch_core_headers INTERFACE)
target_include_directories(poptorch_core_headers INTERFACE ${PROJECT_SOURCE_DIR}/poptorch/include)

set(POPTORCH_CORE_SOURCES
  "poptorch.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/ImplicitCasting.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/LowerToPopart.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/OpBuilder.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/OverlappedIO.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ActivationOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/AtenHandlers.gen.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ArithmeticOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/BilinearOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/BitwiseOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/BlasOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ConstantOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ConvolutionOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/CustomOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/DistanceOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/DropoutOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/EinsumOp.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/EmbeddingOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/IndexOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/LossOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/NormalizationOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/OtherOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/PoolingOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/PopartCanonicalizationUtils.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/PoptorchHandlers.gen.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/PyGTorchScatterOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/RandomSamplingOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ReduceOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ReshapeOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/RNNOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/SliceOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/SoftmaxOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/TensorOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/AddDetachOperations.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/AddSubgraphConnectionNodes.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/AliasProcessing.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/AutomaticCasting.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/CanonicalizeLists.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/CPUOffloadingCleanUp.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/FixupSetAvailableMemory.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/GatherWithExpandedIndicesOptimization.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/InplaceOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PopartCanonicalization.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PopartLateCanonicalization.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PoplarExecutable.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PoptorchSymbols.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/RemoveSurplusIdentityLosses.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/ScatterAddOptimization.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/type_and_constant_canonicalization/AddListNumElements.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/type_and_constant_canonicalization/CanonicaliseConstants.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/type_and_constant_canonicalization/CastUnsupportedInputs.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/type_and_constant_canonicalization/CanonicalizeHalf.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/type_and_constant_canonicalization/CheckAndChangeOutputTypes.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/type_and_constant_canonicalization/EvaluateConstexprs.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/type_and_constant_canonicalization/MakeConstantIntParams.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/Utils.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/ErrorOnUnsupportedAten.cpp"
)

# Same as pybind11_add_module but without stripping the symbols and setting the visibility to hidden.
# Source: https://pybind11.readthedocs.io/en/stable/compiling.html#advanced-interface-library-targets
#
# If the symbols are stripped then error messages will only contain symbol
# addresses instead of human readable names.
add_library(poptorch_core MODULE
  ${POPTORCH_CORE_SOURCES}
  )
pybind11_extension(poptorch_core)
# skip pybind11_strip(target)

target_link_libraries(poptorch_core PRIVATE
  pybind11::module
  pybind11::lto
  popart_compiler
  torch
  ${TORCH_PYTHON_LIBRARY}
  poptorch_err
  poptorch_logging
  dispatch_tracer
  poptorch_core_headers
  stdc++fs
)

install(TARGETS poptorch_core
  DESTINATION ${INSTALL_PYDIR})

file(GLOB python_files "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

# __init__.py needs to be edited by set_version.py so don't copy it over.
list(REMOVE_ITEM python_files "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py")

run_poptorch_install_command("python3 ${PROJECT_SOURCE_DIR}/scripts/set_version.py --torch-version ${TORCH_VERSION} ${CMAKE_CURRENT_BINARY_DIR}/__init__.py" "${PROJECT_SOURCE_DIR}" "Generate __init__.py")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py DESTINATION "${INSTALL_PYDIR}")
install(FILES ${python_files}  py.typed DESTINATION "${INSTALL_PYDIR}")
