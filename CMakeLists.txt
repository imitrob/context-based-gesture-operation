cmake_minimum_required(VERSION 3.0.2)
project(context_based_gesture_operation)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  actionlib_msgs
  geometry_msgs
  sensor_msgs
)

## Generate messages in the 'msg' folder
 add_message_files(
   FILES
   SceneObject.msg
   Scene.msg
   Intent.msg
   Gestures.msg
 )

## Generate services in the 'srv' folder
add_service_files (
  FILES
  G2I.srv
)

include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

## Generate actions in the 'action' folder
add_action_files(
 FILES
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  actionlib_msgs
  sensor_msgs
)

## C++ libs & executables

catkin_package(
  CATKIN_DEPENDS message_runtime
  )
