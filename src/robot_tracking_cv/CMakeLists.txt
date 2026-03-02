cmake_minimum_required(VERSION 3.8)
project(robot_tracking_cv)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(tracker_node src/tracker_node.cpp)
ament_target_dependencies(tracker_node rclcpp sensor_msgs geometry_msgs cv_bridge OpenCV)

install(TARGETS tracker_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()