cmake_minimum_required(VERSION 3.15)
project(vis_ear)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 14)

find_package(PCL 1.9 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)
# Eigen
include_directories("/usr/include/eigen3")

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (vis_ear  main.cpp)
target_link_libraries (vis_ear ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Pangolin_LIBRARIES})






