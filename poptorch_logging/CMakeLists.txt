cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(poptorch_logging)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(spdlog 1.8.0 EXACT REQUIRED)

# Packages provided by Poplar
find_package(libpvti REQUIRED)
find_package(gccs REQUIRED)

add_library(poptorch_logging STATIC
  "source/Error.cpp"
  "source/Logging.cpp"
  "source/Tracepoint.cpp")

file(GLOB_RECURSE poptorch_logging_public_headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp*")

set_target_properties(poptorch_logging PROPERTIES
  CXX_STANDARD 14
  PUBLIC_HEADER "${poptorch_logging_public_headers}")

target_include_directories(poptorch_logging SYSTEM
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# Unfortunately, there seems to be an issue with using the `spdlog::*` targets
# directly with `target_link_libraries()`, which breaks dependencies of
# `poptorch_logging` adding any other include directories. Instead, we'll
# manually add spdlog's include directories and compile definitions here.
target_include_directories(poptorch_logging SYSTEM
  PUBLIC
    $<TARGET_PROPERTY:spdlog::spdlog_header_only,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(poptorch_logging
  PUBLIC
    $<TARGET_PROPERTY:spdlog::spdlog_header_only,INTERFACE_COMPILE_DEFINITIONS>)

target_link_libraries(poptorch_logging
  PRIVATE
    libpvti
    gccs_stacktrace)

install(TARGETS poptorch_logging
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/poptorch_logging)
