cmake_minimum_required(VERSION 3.15)
project(mppi_cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(
    mppi_cpp 
    src/mppi/cpp/mppi_cpp.cpp 
    src/mppi/cpp/bindings.cpp
)

set_target_properties(mppi_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/mppi/cpp
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/mppi/cpp
)