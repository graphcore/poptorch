cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(poptorch_logging)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(spdlog 1.8.0 EXACT REQUIRED)

add_library(poptorch_logging STATIC
  "source/Error.cpp"
  "source/Logging.cpp")

set_property(TARGET poptorch_logging PROPERTY CXX_STANDARD 14)

target_include_directories(poptorch_logging SYSTEM PUBLIC
                                        include
                                        )
target_link_libraries(poptorch_logging spdlog::spdlog_header_only)
