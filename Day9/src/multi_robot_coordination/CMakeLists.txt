cmake_minimum_required(VERSION 3.5)
project(multi_robot_coordination)

# Required packages
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CylinderDetection.msg"
  DEPENDENCIES std_msgs
)

# Ensure message dependencies are exported
ament_export_dependencies(rosidl_default_runtime)

# Package definition
ament_package()

