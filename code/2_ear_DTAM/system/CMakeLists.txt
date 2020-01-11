cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)

set(TARGET_NAME example)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(STDFS_DIRS c++fs)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(STDFS_DIRS stdc++fs)
endif ()

include(FetchContent)
FetchContent_Declare(
        cnpy
        GIT_REPOSITORY https://github.com/rogersce/cnpy.git
)
FetchContent_MakeAvailable(cnpy)

add_executable(${TARGET_NAME}
        main.cpp
        System.cpp
        System.hpp
        Dataset.cpp
        Dataset.hpp
        )

target_include_directories(${TARGET_NAME}
        SYSTEM PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${cnpy_SOURCE_DIR})

message("${ZLIB_LIBRARIES}")
target_link_libraries(${TARGET_NAME}
        PRIVATE
        dtam_core
        Eigen3::Eigen
        ${STDFS_DIRS}
        ${OpenCV_LIBS}
        cnpy
        )

# HighWarning(${TARGET_NAME})