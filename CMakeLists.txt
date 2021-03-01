cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(poptorch)

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

# Note: The next line is also parsed by scripts/utils/_utils.py
set(TORCH_VERSION 1.7.1)

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

function(create_symlink_to_include target linkname)
  get_target_property(dirs ${target} INTERFACE_INCLUDE_DIRECTORIES)
  list(GET dirs 0 include_dir)
  file(CREATE_LINK ${include_dir} ${linkname} SYMBOLIC)
endfunction()

create_symlink_to_include(poplar poplar)
create_symlink_to_include(popart-only popart)

get_target_property(POPLAR_LIB poplar LOCATION)
get_filename_component(POPLAR_DIR ${POPLAR_LIB} DIRECTORY)
# Run an install command that requires PopTorch, PopArt and Poplar to be in the PATH.
function(run_poptorch_install_command cmd working_directory cmd_name)
  install(CODE
    "set(ENV{LD_LIBRARY_PATH} ${popart_LIB_DIR}:${POPLAR_DIR}:$ENV{LD_LIBRARY_PATH})
    set(ENV{POPTORCH_SMALL_IPU_MODEL} 1)
    execute_process( COMMAND ${cmd} WORKING_DIRECTORY ${working_directory} RESULT_VARIABLE RETVAL )
    if(RETVAL AND NOT RETVAL EQUAL 0)
      message(FATAL_ERROR \"${cmd_name} FAILED (See above)\")
    endif()")
endfunction()

file(WRITE ${CMAKE_BINARY_DIR}/tmp/test.sh "#!$ENV{SHELL}
  source ${CMAKE_BINARY_DIR}/enable.sh
  ctest --output-on-failure --timeout 3600 $@
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
  -Wuninitialized)

if(${ENABLE_WERROR})
  add_compile_options(-Werror)
endif()

set(CMAKE_CXX_STANDARD 17)

set(INSTALL_PYDIR poptorch)

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

add_subdirectory(poptorch_logging)
add_subdirectory(popart_compiler)
add_subdirectory(python)

# Examples and tests
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(docs/user_guide)

add_custom_target(poptorch_standalone_wheel
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
  COMMAND  ${CMAKE_COMMAND} -E env WHEEL_LIB_DIRS="${popart_LIB_DIR}:${POPLAR_DIR}" python3 ${CMAKE_CURRENT_SOURCE_DIR}/setup.py sdist bdist_wheel
  DEPENDS poptorch_core
)

add_custom_target(poptorch_wheel
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/setup.py sdist bdist_wheel
  DEPENDS poptorch_core
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
      DEPENDS poptorch_core
  )
endif()
