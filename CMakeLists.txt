cmake_minimum_required(VERSION 3.18)
project(perception_pipeline VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

set(ONNXRUNTIME_ROOT "/d/onnxruntime/onnxruntime-win-x64-1.19.2" CACHE PATH "ONNX Runtime root")

if(EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(STATUS "ONNX Runtime found at: ${ONNXRUNTIME_ROOT}")
    set(HAS_ONNX TRUE)
    set(ORT_INCLUDE "${ONNXRUNTIME_ROOT}/include")
    set(ORT_LIB     "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
else()
    message(WARNING "ONNX Runtime not found - detector will use stub mode")
    set(HAS_ONNX FALSE)
endif()

set(PIPELINE_SOURCES
    src/main.cpp
    src/calibration/calibration.cpp
    src/preprocess/preprocess.cpp
    src/depth/depth_estimator.cpp
    src/detector/detector.cpp
    src/tracker/kalman_tracker.cpp
    src/fusion/fusion.cpp
    src/renderer/renderer.cpp
    src/optical_flow/optical_flow.cpp
    src/pose/pose_estimator.cpp
    src/pipeline/threaded_pipeline.cpp
)

add_executable(${PROJECT_NAME} ${PIPELINE_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)

if(HAS_ONNX)
    target_include_directories(${PROJECT_NAME} PRIVATE ${ORT_INCLUDE})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ORT_LIB})
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAS_ONNX)
endif()

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

message(STATUS "OpenCV version : ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime   : ${HAS_ONNX}")
message(STATUS "Build type     : ${CMAKE_BUILD_TYPE}")
