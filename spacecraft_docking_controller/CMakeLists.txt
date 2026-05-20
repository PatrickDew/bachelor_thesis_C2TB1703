cmake_minimum_required(VERSION 3.8)
project(spacecraft_docking_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Custom message generation
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SpacecraftState.msg"
  "msg/ControlCommand.msg"
  "msg/DockingStatus.msg"
  "msg/ControllerGains.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Install Python modules
ament_python_install_package(sdc_core)

# Install Python executables
install(PROGRAMS
  scripts/docking_controller_node
  scripts/pid_controller_node
  scripts/lqr_controller_node
  scripts/mpc_controller_node
  scripts/state_estimator_node
  scripts/pose_array_bridge_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install resource marker
install(FILES
  resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

