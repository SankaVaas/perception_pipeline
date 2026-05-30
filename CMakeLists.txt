cmake_minimum_required(VERSION 3.18)
project(perception_pipeline VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

set(PIPELINE_SOURCES
    src/main.cpp
    src/calibration/calibration.cpp
    src/preprocess/preprocess.cpp
    src/depth/depth_estimator.cpp
    src/detector/detector.cpp
    src/tracker/kalman_tracker.cpp
    src/fusion/fusion.cpp
    src/renderer/renderer.cpp
)

add_executable(${PROJECT_NAME} ${PIPELINE_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "OpenCV version : ${OpenCV_VERSION}")
message(STATUS "Build type     : ${CMAKE_BUILD_TYPE}")
