set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

find_package(OpenCV REQUIRED)


#set(CMAKE_BUILD_TYPE Debug)
#Release

include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set (CMAKE_CXX_STANDARD 11)

set(SOURCES src/EnvConfig.cpp src/Environment.cpp src/Attention.cpp)
# Build our plugin
# Build the stand-alone test program
add_executable(search ${SOURCES})
target_link_libraries(search ${OpenCV_LIBRARIES})



if(CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES)
   list(REMOVE_DUPLICATES CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES)
endif(CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES)
