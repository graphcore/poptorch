cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(poptorch)

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

link_directories("${POPART_INSTALL}/lib")


add_subdirectory(popart_compiler)

pybind11_add_module(poptorch_core SHARED poptorch.cpp "source/LowerToPopart.cpp")

# Enable C++11
target_compile_features(poptorch_core PRIVATE cxx_range_for)

target_link_libraries(poptorch_core PUBLIC
                                     ${TORCH_LIBRARIES}
                                     popart_compiler)

target_include_directories(poptorch_core PRIVATE
                                        "include"
                                        "popart_compiler/include")


configure_file(__init__.py.in poptorch/__init__.py)
