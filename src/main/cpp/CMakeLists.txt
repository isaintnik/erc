cmake_minimum_required(VERSION 3.10)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(pybind11)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../python)
pybind11_add_module(wheel wheel.cpp)
