cmake_minimum_required(VERSION 3.4.1)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/include)

add_library(lib_opencv SHARED IMPORTED)
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopencv_java4.so)

add_library(lib_onnxruntime SHARED IMPORTED)
set_target_properties(lib_onnxruntime PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libonnxruntime.so)

set(EDGE_DETECTION_DIR "../ios/Classes")
set(SOURCES
    ${EDGE_DETECTION_DIR}/native_edge_detection.cpp
    ${EDGE_DETECTION_DIR}/edge_detector.cpp
    ${EDGE_DETECTION_DIR}/image_processor.cpp
    ${EDGE_DETECTION_DIR}/new_backend/ffi_exports.cpp
    ${EDGE_DETECTION_DIR}/new_backend/SpotDetector.cpp
    ${EDGE_DETECTION_DIR}/new_backend/RFCalculator.cpp
    ${EDGE_DETECTION_DIR}/new_backend/AUCCalculator.cpp
)

add_library(native_edge_detection SHARED ${SOURCES})

# Disable strict MSVC-only attributes for clang/NDK build
target_compile_definitions(native_edge_detection PRIVATE -D__declspec\(x\)=)

target_link_libraries(native_edge_detection lib_opencv lib_onnxruntime)
find_library(log-lib log)
target_link_libraries(native_edge_detection ${log-lib})