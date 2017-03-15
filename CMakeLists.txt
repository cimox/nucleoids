cmake_minimum_required(VERSION 3.6)
project(nucleoids)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )
set(SOURCE_FILES src/main.cpp)
add_executable(nucleoids ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(nucleoids ${OpenCV_LIBS})