cmake_minimum_required(VERSION 3.6)
project(nucleoids)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
set(SOURCE_FILES src/main.cpp src/common/utils.cpp src/common/utils.h src/core/draw.h src/core/draw.cpp)
add_executable(nucleoids ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(nucleoids ${OpenCV_LIBS})