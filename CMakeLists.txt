cmake_minimum_required(VERSION 3.8)
project(dcnn)

set(CMAKE_CXX_STANDARD 17)

include_directories(src)

add_executable(dcnn src/main.cpp)