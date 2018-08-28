set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})


find_package(roscpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(custom_msg REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
)


#set(CMAKE_BUILD_TYPE Debug)
#Release
include_directories(${roscpp_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS} ${sensor_msgs_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${custom_msg_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  ${roscpp_CXX_FLAGS}")
set (CMAKE_CXX_STANDARD 11)

set(SOURCES src/EnvConfig.cpp src/Environment.cpp src/Attention.cpp)
# Build our plugin
# Build the stand-alone test program
add_executable(search ${SOURCES})
target_link_libraries(search ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${roscpp_LIBRARIES} ${std_msgs_LIBRARIES} ${catkin_LIBRARIES})



if(CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES)
   list(REMOVE_DUPLICATES CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES)
endif(CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES)
