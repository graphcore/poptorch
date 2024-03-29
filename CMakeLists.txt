cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(poptorch)
include(GNUInstallDirs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Default to local install prefix" FORCE)
endif()

set(USE_PYTORCH_PACKAGE_HEADERS ON CACHE BOOL "Use the Torch headers distributed with the pytorch package.")
set(POPLAR_DIR CACHE PATH "Path to a Poplar install")
set(POPART_DIR CACHE PATH "Path to a Popart install")
set(SNAPSHOT "" CACHE STRING "Snapshot ID to use for the documentation")
set(SDK_DIR CACHE PATH "Path to an extracted SDK archive or to a Poplar & Popart install directory (Will populate POPLAR_DIR and POPART_DIR)")
set(BUILD_DOCS OFF CACHE BOOL "Build PopTorch's documentation")
set(COPY_TESTS OFF CACHE BOOL "Copy tests files to the build folder (instead of running them from the sources folder)")
set(ENABLE_WERROR ON CACHE BOOL "Treat C++ warnings as errors")
set(EXTRA_PYTEST_ARGS "" CACHE STRING "Extra arguments to pass to pytest when generating the list of tests to run")

# Always use the gold linker to avoid segfaults with PopART / Poplar on some OSes.
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  find_program(GOLD_EXECUTABLE ld.gold REQUIRED)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=gold")
endif()

# Note: The next line is also parsed by scripts/utils/_utils.py
set(TORCH_VERSION 2.0.1)

# Convert to cmake list
string(REPLACE "." ";" TORCH_VERSION_AS_LIST ${TORCH_VERSION})

# Get the minor component. (Versions are Major.Minor.Patch)
list(GET TORCH_VERSION_AS_LIST 1 TORCH_MINOR_VERSION)

set(CMAKE_CONFIGURATION_TYPES "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
if(NOT CMAKE_BUILD_TYPE)
  list(GET CMAKE_CONFIGURATION_TYPES 0 CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified")
endif()
if(NOT CMAKE_BUILD_TYPE IN_LIST CMAKE_CONFIGURATION_TYPES)
  message(FATAL_ERROR "CMAKE_BUILD_TYPE must be one of ${CMAKE_CONFIGURATION_TYPES}")
endif()

if(USE_PYTORCH_PACKAGE_HEADERS)
  execute_process(COMMAND python3 -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent, end='')"
    OUTPUT_VARIABLE TORCH_PATH)
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
  if(NOT TORCH_PATH)
    message(FATAL_ERROR "python3 -c \"import torch\" failed: check that your virtual environment is active and torch is installed")
  endif()
  execute_process(COMMAND python3 -c "import torch; import sys; sys.exit(3 if torch.version.debug else 4)"
    RESULT_VARIABLE TORCH_DEBUG)
  if(TORCH_DEBUG LESS 3 OR TORCH_DEBUG GREATER 4)
    message(FATAL_ERROR "python3 -c \"import torch\" failed: check that your virtual environment is active and torch is installed")
  endif()
  if (TORCH_DEBUG EQUAL 4)
    # We include torch header files which respond to this flag, so we have to
    # set it correctly in order to get consistent torch behaviour.
    add_compile_definitions(NDEBUG)
  endif()
endif()

if(EXISTS ${SDK_DIR})
  execute_process(COMMAND find ${SDK_DIR} -maxdepth 1 -type d -name "popart*"
    OUTPUT_VARIABLE POPART_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND find ${SDK_DIR} -maxdepth 1 -type d -name "poplar-*" -o -name "poplar"
    OUTPUT_VARIABLE POPLAR_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT IS_DIRECTORY "${POPLAR_DIR}")
    message(FATAL_ERROR "Couldn't find a \"poplar\" or \"poplar-*\" folder in '${SDK_DIR}'")
  endif()
  if(NOT IS_DIRECTORY "${POPART_DIR}")
    message(FATAL_ERROR "Couldn't find a \"popart*\" folder in '${SDK_DIR}'")
  endif()
endif()

if(EXISTS ${POPLAR_DIR})
  list(APPEND CMAKE_PREFIX_PATH ${POPLAR_DIR})
  set(ENABLE_POPLAR_CMD "source ${POPLAR_DIR}/enable.sh")
else()
  # Check the package is not already in the path
  find_package(poplar)
  if(NOT poplar_FOUND)
    message(FATAL_ERROR "You must provide a path to a Poplar install using -DPOPLAR_DIR=/path/to/popart/build/install")
  endif()
endif()

if( EXISTS ${POPART_DIR} )
  list(APPEND CMAKE_PREFIX_PATH ${POPART_DIR})
  set(ENABLE_POPART_CMD "source ${POPART_DIR}/enable.sh")
else()
  find_package(popart COMPONENTS popart-only)
  if(NOT popart_FOUND)
    message(FATAL_ERROR "You must provide a path to a Popart build using -DPOPART_DIR=/path/to/popart/build")
  endif()
endif()

if(NOT popart_FOUND)
  find_package(popart REQUIRED COMPONENTS popart-only)
endif()
if(NOT poplar_FOUND)
  find_package(poplar REQUIRED)
endif()

get_target_property(POPLAR_LIB poplar LOCATION)
get_filename_component(POPLAR_DIR ${POPLAR_LIB} DIRECTORY)
# Run an install command that requires PopTorch, PopArt and Poplar to be in the PATH.
function(run_poptorch_install_command cmd working_directory cmd_name)
  install(CODE
    "set(ENV{LD_LIBRARY_PATH} ${popart_LIB_DIR}:${POPLAR_DIR}:$ENV{LD_LIBRARY_PATH})
    set(ENV{POPTORCH_SMALL_IPU_MODEL} 1)
    execute_process( COMMAND ${cmd} WORKING_DIRECTORY ${working_directory} RESULT_VARIABLE RETVAL OUTPUT_VARIABLE OUTPUT ERROR_VARIABLE OUTPUT)
    if(RETVAL AND NOT RETVAL EQUAL 0)
      message(FATAL_ERROR \"${cmd_name} FAILED: \${OUTPUT}\")
    endif()")
endfunction()

function(remove_use_distributed_definition target)
  get_target_property(compile_options ${target} INTERFACE_COMPILE_DEFINITIONS)
  if(NOT compile_options)
    message(FATAL_ERROR "Could not get property INTERFACE_COMPILE_DEFINITIONS from target '${target}'.")
  endif()
  list(REMOVE_ITEM compile_options USE_DISTRIBUTED)
  list(REMOVE_ITEM compile_options USE_RPC)
  set_target_properties(${target} PROPERTIES INTERFACE_COMPILE_DEFINITIONS "${compile_options}")
endfunction()

file(WRITE ${CMAKE_BINARY_DIR}/tmp/test.sh "#!$ENV{SHELL}
  source ${CMAKE_BINARY_DIR}/enable.sh
  export POPTORCH_TEST_TIMEOUT=3600
  export TORCH_SHOW_CPP_STACKTRACES=1
  ctest --output-on-failure --timeout $POPTORCH_TEST_TIMEOUT $@
  ")
file(COPY ${CMAKE_BINARY_DIR}/tmp/test.sh
  DESTINATION ${CMAKE_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

enable_testing()

add_compile_options(
  -Wall
  -pedantic
  -Wextra
  -Wdisabled-optimization
  -Wshadow
  -Wswitch
  -Wformat=2
  -Wimplicit-fallthrough
  -Winit-self
  -Wcomment
  -Wsequence-point
  -Wundef
  -Wuninitialized
  -DTORCH_MINOR_VERSION=${TORCH_MINOR_VERSION})

if(${ENABLE_WERROR})
  add_compile_options(-Werror)
endif()

set(CMAKE_CXX_STANDARD 17)

set(INSTALL_PYDIR ${CMAKE_INSTALL_PREFIX}/poptorch)

if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
  set(CMAKE_INSTALL_RPATH "@loader_path")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
else()
  # $ORIGIN/lib is needed by the standalone wheel: by default libraries expect
  # their dependencies to be in $ORIGIN/../lib therefore for Poplar
  # and Popart to work they need to be stored in a folder named "lib".
  # However the poptorch shared libraries which are loaded from python must be at
  # the root of the poptorch package, which is why we need to add $ORIGIN/lib
  # to the poptorch libraries.
  set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/lib")
endif()

find_package(Torch ${TORCH_VERSION} EXACT REQUIRED)

remove_use_distributed_definition(torch_cpu)

add_subdirectory(poptorch_err)
add_subdirectory(poptorch_logging)
add_subdirectory(poptorch_compiler/pytorch_bridge)

add_subdirectory(poptorch/source/dispatch_tracer)
add_subdirectory(popart_compiler)
add_subdirectory(poptorch)
add_subdirectory(python)
add_subdirectory(poptorch_geometric)

# Examples and tests
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(docs/user_guide)

add_custom_target(poptorch_standalone_wheel
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
  COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/generate_python_package.py bdist_wheel --include-dir ${CMAKE_INSTALL_PREFIX}/include --lib-dir ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} --output-dir ${CMAKE_INSTALL_PREFIX}/dist --python-dir ${INSTALL_PYDIR} --standalone "${popart_LIB_DIR}:${POPLAR_DIR}"
  DEPENDS poptorch
)
add_custom_target(poptorch_standalone_sdist
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
  COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/generate_python_package.py sdist --include-dir ${CMAKE_INSTALL_PREFIX}/include --lib-dir ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} --output-dir ${CMAKE_INSTALL_PREFIX}/dist --python-dir ${INSTALL_PYDIR} --standalone "${popart_LIB_DIR}:${POPLAR_DIR}"
  DEPENDS poptorch
)

add_custom_target(poptorch_wheel
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
  COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/generate_python_package.py bdist_wheel --include-dir ${CMAKE_INSTALL_PREFIX}/include --lib-dir ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} --output-dir ${CMAKE_INSTALL_PREFIX}/dist --python-dir ${INSTALL_PYDIR}
  DEPENDS poptorch
)

add_custom_target(poptorch_sdist
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
  COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/generate_python_package.py sdist --include-dir ${CMAKE_INSTALL_PREFIX}/include --lib-dir ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} --output-dir ${CMAKE_INSTALL_PREFIX}/dist --python-dir ${INSTALL_PYDIR}
  DEPENDS poptorch
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md
        DESTINATION .)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/enable.sh.in
  ${PROJECT_BINARY_DIR}/enable.sh
  @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/enable.sh
        DESTINATION .)

if(${CMAKE_VERSION} VERSION_GREATER "3.15.0")
  # Building poptorch without installing it doesn't make sense: the python
  # module cannot be used so always install after a build.
  add_custom_target(install_poptorch ALL
      COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR}
      DEPENDS poptorch custom_cube_op custom_leaky_relu_op custom_add_scalar_op custom_add_scalar_vec_op custom_add_vec_scalar_mul_op custom_reduce_op custom_three_input_reduce_op custom_many_attribute_op
  )
endif()
