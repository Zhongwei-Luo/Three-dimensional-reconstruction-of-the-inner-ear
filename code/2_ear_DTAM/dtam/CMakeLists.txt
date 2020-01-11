cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB)

include(FetchContent)
FetchContent_Declare(
        cnpy
        GIT_REPOSITORY https://github.com/rogersce/cnpy.git
)
FetchContent_MakeAvailable(cnpy)

set(TARGET_NAME dtam_core)

add_library(${TARGET_NAME} STATIC
        include/DenseMapper.hpp
        include/Frame.hpp
        include/Camera.hpp
        include/SparseTracker.hpp
        include/utils/MatrixType.hpp
		include/utils/ResultWritingUtil.hpp
        src/Frame.cpp
        src/dense_mapper/DenseMapper.cpp
        src/dense_mapper/cost_volume/CostVolume.cpp
        src/dense_mapper/cost_volume/CostVolume.hpp
        src/dense_mapper/optimizer/Optimizer.cpp
        src/dense_mapper/optimizer/Optimizer.hpp
        src/dense_mapper/utils/MathUtils.hpp
        src/sparse_tracker/SparseTracker.cpp
        src/sparse_tracker/feature_handler/FeatureHandler.hpp
        src/sparse_tracker/feature_handler/FeatureHandler.cpp
        src/sparse_tracker/feature_handler/ORBBFFeatureHandler.cpp
        src/sparse_tracker/utils/ScalePropagationUtil.hpp
        src/sparse_tracker/utils/ScalePropagationUtil.cpp
        src/sparse_tracker/utils/CommonUtil.hpp
        src/sparse_tracker/utils/RelativePoseUtil.hpp
        src/sparse_tracker/utils/CommonUtil.cpp
        src/sparse_tracker/utils/RelativePoseUtil.cpp
        )

target_include_directories(${TARGET_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
        src
		)

target_include_directories(${TARGET_NAME}
        SYSTEM PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${cnpy_SOURCE_DIR}
		)

target_link_libraries(${TARGET_NAME}
        PRIVATE
        Eigen3::Eigen
        ${OpenCV_LIBS}
		cnpy
		)

# HighWarning(${TARGET_NAME})
