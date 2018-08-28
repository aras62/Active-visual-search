cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11")
project(saliency)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
  cv_bridge
)

# Hack for ROS Kinetic and Ubuntu 16.04
SET("OpenCV_DIR" "/opt/ros/kinetic/")
#find_package(OpenCV REQUIRED)

## Generate services in the 'srv' folder
 add_service_files(
  FILES
  GetAIM.srv
  GetBackProj.srv
 )


## Generate added messages and services with any dependencies listed here 
generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
 )
catkin_package(
CATKIN_DEPENDS message_runtime
)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${roscpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  ${roscpp_CXX_FLAGS}")
set (CMAKE_CXX_STANDARD 11)
set(SOURCES src/Saliency.cpp)
add_executable(saliency ${SOURCES})

target_link_libraries(saliency ${Boost_LIBRARIES} ${OpenCV_LIBS} ${roscpp_LIBRARIES} ${std_msgs_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${catkin_LIBRARIES})


