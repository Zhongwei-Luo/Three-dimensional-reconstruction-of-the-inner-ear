cmake_minimum_required(VERSION 3.14)
project(DTAM)
set(CMAKE_BUILD_TYPE  Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/WarningLevel.cmake)

add_subdirectory(dtam)
add_subdirectory(system)