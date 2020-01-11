cmake_minimum_required(VERSION 3.15)
project(opencv_reproject)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Debug")




find_package(OpenCV 3 REQUIRED)



include_directories(
        ${OpenCV_INCLUDE_DIRS}

        "/usr/local/include/eigen3/"
)




add_executable(opencv_reproject main.cpp)
target_link_libraries(opencv_reproject

        ${OpenCV_LIBS} )

