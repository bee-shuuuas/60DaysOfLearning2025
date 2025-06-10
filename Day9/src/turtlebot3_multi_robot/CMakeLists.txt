################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 3.5)
project(turtlebot3_multi_robot)

################################################################################
# Find ament packages and libraries for ament and system dependencies
################################################################################
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)

################################################################################
# Install
################################################################################
install(DIRECTORY launch models rviz urdf worlds params
  DESTINATION share/${PROJECT_NAME}/
)

################################################################################
# Install Python executables
################################################################################
install(PROGRAMS
  node/random_walker.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  node/grid_based_explorer.py
  DESTINATION lib/${PROJECT_NAME}
)
  
install(PROGRAMS
  node/frontier_based_exploration.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  node/multi_robot_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  node/obstacle_avoidance.py
  DESTINATION lib/${PROJECT_NAME}
)


ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/multi_robot.dsv.in")

ament_package()