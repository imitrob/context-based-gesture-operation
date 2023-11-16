cmake_minimum_required(VERSION 3.5)
project(context_based_gesture_operation)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Gestures.msg"
  "msg/Intent.msg"
  "msg/Scene.msg"
  "msg/SceneObject.msg"
  "msg/HRICommand.msg"
  "srv/G2I.srv"
  "srv/BTreeSingleCall.srv"
  DEPENDENCIES geometry_msgs std_msgs sensor_msgs # Add packages that above messages depend on, in this case geometry_msgs for Sphere.msg
)

# Install Python package
ament_python_install_package(context_based_gesture_operation)

# Install examples
set(EXAMPLES_DIR context_based_gesture_operation)
install(PROGRAMS
    ${EXAMPLES_DIR}/srcmodules/BTreeLib.py
    ${EXAMPLES_DIR}/agent_nodes/btree.py
    DESTINATION lib/${PROJECT_NAME}
)

#ament_export_dependencies(rosidl_default_runtime)

# Setup the project
ament_package()
