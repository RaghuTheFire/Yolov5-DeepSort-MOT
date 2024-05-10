cmake_minimum_required(VERSION 3.5)

project(DeepSORT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# set(ONNXRUNTIME_DIR "/home/a/lib/onnxruntime-linux-x64-1.12.1")
# include_directories("${ONNXRUNTIME_DIR}/include")
find_package(OpenCV 4 REQUIRED )

include_directories(
    ${OpenCV_INCLUDE_DIRS}/include
    ${CMAKE_SOURCE_DIR}/inc
    )
	
add_executable(DeepSORT
    /src/YOLOv5Detector.cpp
    /src/FeatureTensor.cpp
    /src/model.cpp
    /src/kalmanfilter.cpp
    /src/linear_assignment.cpp
    /src/nn_matching.cpp
    /src/track.cpp
    /src/tracker.cpp
    /src/munkres.cpp
    /src/hungarianoper.cpp	
    DeepSort_Main.cpp)

# target_link_libraries(DeepSORT PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" ${OpenCV_LIBS} Eigen3::Eigen)

target_link_libraries(DeepSORT PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)

add_executable(test_detector
    /src/YOLOv5Detector.cpp
    Test_Detector.cpp)
target_link_libraries(test_detector PRIVATE  ${OpenCV_LIBS} Eigen3::Eigen)
