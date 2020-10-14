cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(poptorch_logging)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(spdlog REQUIRED)

add_library(poptorch_logging STATIC
  "source/Error.cpp"
  "source/Logging.cpp")

set_property(TARGET poptorch_logging PROPERTY CXX_STANDARD 14)

target_include_directories(poptorch_logging SYSTEM PUBLIC
                                        ${SPDLOG_INCLUDE_DIR}
                                        include
                                        )
